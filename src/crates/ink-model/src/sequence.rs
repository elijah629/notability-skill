//! Whole-sample text-conditioned cubic sequence training. No inferred glyph labels.
use anyhow::{Context, Result, ensure};
use ink_corpus::{
    Sample, audit, file_hash, load, save,
    sequence::{self, END, FEATURES},
};
use ink_neural::{
    Linear,
    sequence::{
        ALIGN_INPUT, Alignment, EMBED, Gru, HIDDEN, MIXTURES, PARAMETERS, STYLE, SequenceDecoder,
        WINDOWS, decoder_input_width,
    },
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    time::Instant,
};
use tch::{Device, Kind, Tensor, nn, nn::RNN};
const RECIPE: &str = include_str!("../../../recipes/sequence-cvae-v14.json");
const CONTENT_WARMUP: u64 = 10_000;
const ROLLOUT_RAMP: u64 = 20_000;
fn rollout_probability(step: u64) -> f64 {
    (step.saturating_sub(CONTENT_WARMUP) as f64 / ROLLOUT_RAMP as f64).min(1.)
}
fn trajectory_geometry(
    predicted: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    source_height_scale: &Tensor,
) -> Tensor {
    let error = (predicted - target) * source_height_scale;
    (error
        .smooth_l1_loss(&Tensor::zeros_like(&error), tch::Reduction::None, 0.1)
        .mean_dim(&[-1_i64][..], false, Kind::Float)
        * mask)
        .sum(Kind::Float)
        / mask.sum(Kind::Float).clamp_min(1.)
}
struct LearnedAlignment {
    recurrent: nn::GRU,
    window: nn::Linear,
}
struct Network {
    alignment: Option<LearnedAlignment>,
    store: nn::VarStore,
    embed: nn::Embedding,
    encoder: nn::GRU,
    posterior: nn::GRU,
    mu: nn::Linear,
    logvar: nn::Linear,
    decoder: nn::GRU,
    query: nn::Linear,
    out: nn::Linear,
    length: nn::Linear,
    attention_strength: f64,
    local_conditioning: bool,
    token_space_attention: bool,
    global_conditioning: bool,
}
struct Batch {
    tokens: Tensor,
    text_mask: Tensor,
    events: Tensor,
    mask: Tensor,
    weights: Tensor,
    previous: Tensor,
    event_weights: Tensor,
    geometry_scale: Tensor,
    position_scale: Tensor,
}
struct LossComponents {
    reconstruction: Tensor,
    kl: Tensor,
    duration: Tensor,
    absolute: Tensor,
    geometry: Tensor,
}
fn local_geometry(selected: &Tensor, b: &Batch) -> Tensor {
    let move_mask = b.events.narrow(-1, 7, 1);
    let curve_mask = b.events.narrow(-1, 8, 1);
    let error = selected - b.events.narrow(-1, 0, 7);
    // Pen lifts are positions, not samples of writing velocity. Their objective
    // must survive arbitrarily many correctly predicted native curve controls.
    let move_error =
        Tensor::cat(&[error.narrow(-1, 0, 2), error.narrow(-1, 6, 1)], -1) * &b.position_scale;
    let curve_error = error * &b.geometry_scale;
    let reduce = |error: &Tensor, weight: &Tensor, beta: f64| {
        (error.smooth_l1_loss(&Tensor::zeros_like(error), tch::Reduction::None, beta) * weight)
            .sum(Kind::Float)
            / (weight.sum(Kind::Float) * error.size()[2]).clamp_min(1e-8)
    };
    let classes = move_mask.sum(Kind::Float).gt(0.).to_kind(Kind::Float)
        + curve_mask.sum(Kind::Float).gt(0.).to_kind(Kind::Float);
    (reduce(&move_error, &move_mask, 0.02)
        + reduce(&curve_error, &(&curve_mask * b.weights.unsqueeze(-1)), 0.1))
        / classes.clamp_min(1.)
}
impl LossComponents {
    fn total(&self) -> Tensor {
        &self.reconstruction + &self.kl + &self.duration + &self.absolute + &self.geometry
    }
}
impl Network {
    pub(super) fn new(device: Device, vocab: usize) -> Self {
        Self::with_architecture(device, vocab, true, true)
    }
    #[cfg(test)]
    fn with_conditioning(device: Device, vocab: usize, global_conditioning: bool) -> Self {
        Self::with_architecture(device, vocab, global_conditioning, false)
    }
    fn with_architecture(
        device: Device,
        vocab: usize,
        global_conditioning: bool,
        learned_alignment: bool,
    ) -> Self {
        let store = nn::VarStore::new(device);
        let p = &store.root();
        let linear = |name, a, b| nn::linear(p / name, a, b, Default::default());
        let mut length = linear("length", HIDDEN as i64, 1);
        tch::no_grad(|| {
            let _ = length.ws.zero_();
            let _ = length.bs.as_mut().unwrap().fill_(12_f64.ln());
        });
        let alignment = learned_alignment.then(|| {
            let mut window = linear("alignment_window", HIDDEN as i64, (3 * WINDOWS) as i64);
            tch::no_grad(|| {
                window.ws *= 0.01;
                let bias = window.bs.as_mut().unwrap();
                let _ = bias.zero_();
                let _ = bias
                    .narrow(0, WINDOWS as i64, WINDOWS as i64)
                    .fill_(4_f64.ln());
                let _ = bias
                    .narrow(0, (2 * WINDOWS) as i64, WINDOWS as i64)
                    .fill_((1_f64 / 12.).ln());
            });
            LearnedAlignment {
                recurrent: nn::gru(
                    p / "aligner",
                    ALIGN_INPUT as i64,
                    HIDDEN as i64,
                    Default::default(),
                ),
                window,
            }
        });
        Self {
            alignment,
            embed: nn::embedding(p / "embed", vocab as i64, EMBED as i64, Default::default()),
            encoder: nn::gru(
                p / "encoder",
                EMBED as i64,
                HIDDEN as i64,
                Default::default(),
            ),
            posterior: nn::gru(
                p / "posterior",
                FEATURES as i64,
                HIDDEN as i64,
                Default::default(),
            ),
            mu: linear("mu", HIDDEN as i64, STYLE as i64),
            logvar: linear("logvar", HIDDEN as i64, STYLE as i64),
            decoder: nn::gru(
                p / "decoder",
                decoder_input_width(global_conditioning) as i64,
                HIDDEN as i64,
                Default::default(),
            ),
            query: linear("query", HIDDEN as i64, HIDDEN as i64),
            out: linear("output", (HIDDEN * 2 + STYLE) as i64, PARAMETERS as i64),
            length,
            attention_strength: 4.,
            local_conditioning: true,
            token_space_attention: true,
            global_conditioning,
            store,
        }
    }
    fn condition(&self, b: &Batch) -> (Tensor, Tensor) {
        let (s, _) = self.encoder.seq(&b.tokens.apply(&self.embed));
        let mean =
            (&s * b.text_mask.unsqueeze(-1)).sum_dim_intlist(&[1_i64][..], false, Kind::Float)
                / b.text_mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        (s, mean)
    }
    fn posterior(&self, b: &Batch) -> (Tensor, Tensor) {
        let (s, _) = self.posterior.seq(&b.events);
        let mean = (&s * b.mask.unsqueeze(-1)).sum_dim_intlist(&[1_i64][..], false, Kind::Float)
            / b.mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        (
            mean.apply(&self.mu),
            mean.apply(&self.logvar).clamp(-8., 3.),
        )
    }
    fn alignment_parameters(&self, b: &Batch, z: &Tensor, mean: &Tensor) -> Option<Tensor> {
        self.alignment.as_ref().map(|a| {
            let t = b.events.size()[1];
            let counts = b
                .text_mask
                .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                .log();
            let input = Tensor::cat(
                &[
                    b.previous.shallow_clone(),
                    mean.unsqueeze(1).expand([-1, t, -1], true),
                    z.unsqueeze(1).expand([-1, t, -1], true),
                    counts.unsqueeze(1).expand([-1, t, -1], true),
                ],
                -1,
            );
            a.recurrent.seq(&input).0.apply(&a.window)
        })
    }
    fn window(&self, parameters: &Tensor, centers: &Tensor, b: &Batch) -> (Tensor, Tensor) {
        let centers = centers.to_kind(Kind::Float);
        let indices = Tensor::arange(b.tokens.size()[1], (Kind::Float, self.store.device()));
        let mass = parameters
            .narrow(-1, 0, WINDOWS as i64)
            .log_softmax(-1, Kind::Float);
        let width = parameters
            .narrow(-1, WINDOWS as i64, WINDOWS as i64)
            .clamp(-2., 4.)
            .exp();
        let log_weights =
            mass.unsqueeze(-1) - width.unsqueeze(-1) * (centers.unsqueeze(-1) - indices).square();
        let scores = log_weights.logsumexp(-2, false).clamp_min(-69.07755279);
        let progress = centers.mean_dim(&[-1_i64][..], true, Kind::Float)
            / b.text_mask
                .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                .unsqueeze(1);
        (progress, scores)
    }
    fn forward(&self, b: &Batch, z: &Tensor) -> Tensor {
        let (states, mean) = self.condition(b);
        self.forward_conditioned(b, z, &states, &mean)
    }
    fn forward_conditioned(&self, b: &Batch, z: &Tensor, states: &Tensor, mean: &Tensor) -> Tensor {
        let t = b.events.size()[1];
        let expanded = z.unsqueeze(1).expand([-1, t, -1], true);
        let clock_progress = Tensor::arange(t, (Kind::Float, self.store.device()))
            .reshape([1, -1, 1])
            / b.mask
                .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                .unsqueeze(-1);
        let learned = self.alignment_parameters(b, z, mean).map(|parameters| {
            let deltas = parameters
                .narrow(-1, (2 * WINDOWS) as i64, WINDOWS as i64)
                .clamp(-8., 2.)
                .exp();
            let deltas = deltas.to_kind(Kind::Double);
            let centers = deltas.cumsum(1, Kind::Double) - deltas.narrow(1, 0, 1);
            self.window(&parameters, &centers, b)
        });
        let progress = learned
            .as_ref()
            .map_or(clock_progress, |(p, _)| p.shallow_clone());
        let token_position = Tensor::arange(b.tokens.size()[1], (Kind::Float, self.store.device()))
            .reshape([1, 1, -1])
            / b.text_mask
                .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                .unsqueeze(-1);
        let units = if self.token_space_attention {
            b.text_mask
                .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                .unsqueeze(-1)
        } else {
            Tensor::ones([1, 1, 1], (Kind::Float, self.store.device()))
        };
        let distance = (&progress - &token_position) * units;
        let recurrent_text = if self.local_conditioning {
            learned
                .as_ref()
                .map_or_else(
                    || -distance.square() * self.attention_strength,
                    |(_, scores)| scores.shallow_clone(),
                )
                .masked_fill(&b.text_mask.eq(0.).unsqueeze(1), -1e9)
                .softmax(-1, Kind::Float)
                .bmm(states)
        } else {
            mean.unsqueeze(1).expand([-1, t, -1], true)
        };
        let mut input = vec![b.previous.shallow_clone(), recurrent_text];
        if self.global_conditioning {
            input.push(mean.unsqueeze(1).expand([-1, t, -1], true));
        }
        input.push(expanded.shallow_clone());
        input.push(progress.shallow_clone());
        if self.global_conditioning {
            input.push(
                b.text_mask
                    .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                    .log()
                    .unsqueeze(1)
                    .expand([-1, t, -1], true),
            );
        }
        let input = Tensor::cat(&input, -1);
        let (h, _) = self.decoder.seq(&input);
        let logits = h.apply(&self.query).bmm(&states.transpose(1, 2)) / (HIDDEN as f64).sqrt()
            + learned.as_ref().map_or_else(
                || -distance.square() * self.attention_strength,
                |(_, scores)| scores.shallow_clone(),
            );
        let attention = logits
            .masked_fill(&b.text_mask.eq(0.).unsqueeze(1), -1e9)
            .softmax(-1, Kind::Float);
        let context = attention.bmm(states);
        Tensor::cat(&[h, context, expanded], -1).apply(&self.out)
    }
    #[cfg(test)]
    fn forward_scheduled(&self, b: &Batch, z: &Tensor, step: u64) -> Tensor {
        let (states, mean) = self.condition(b);
        self.forward_scheduled_conditioned(b, z, step, &states, &mean)
    }
    fn forward_scheduled_conditioned(
        &self,
        b: &Batch,
        z: &Tensor,
        step: u64,
        states: &Tensor,
        mean: &Tensor,
    ) -> Tensor {
        let device = self.store.device();
        let batch_size = b.events.size()[0];
        let duration = b.mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        let expected = if rollout_probability(step) >= 1. {
            (mean.apply(&self.length)
                + b.text_mask
                    .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                    .log())
            .detach()
            .exp()
            .clamp(2., b.events.size()[1] as f64 * 2.)
        } else {
            duration
        };
        let token_position = Tensor::arange(b.tokens.size()[1], (Kind::Float, device)).unsqueeze(0)
            / b.text_mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        let units = if self.token_space_attention {
            b.text_mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float)
        } else {
            Tensor::ones([1, 1], (Kind::Float, device))
        };
        let probability = rollout_probability(step);
        let mut state = self.decoder.zero_state(batch_size);
        let mut alignment_state = self
            .alignment
            .as_ref()
            .map(|a| a.recurrent.zero_state(batch_size));
        let mut centers = Tensor::zeros([batch_size, WINDOWS as i64], (Kind::Double, device));
        let mut prediction = Tensor::zeros([batch_size, FEATURES as i64], (Kind::Float, device));
        let mut output = vec![];
        for t in 0..b.events.size()[1] {
            let use_prediction = if t == 0 {
                Tensor::zeros([batch_size, 1], (Kind::Float, device))
            } else {
                Tensor::rand([batch_size, 1], (Kind::Float, device))
                    .lt(probability)
                    .to_kind(Kind::Float)
            };
            let previous: Tensor =
                b.previous.select(1, t) * (1. - &use_prediction) + &prediction * use_prediction;
            let learned = self.alignment.as_ref().map(|a| {
                let count = b
                    .text_mask
                    .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                    .log();
                let input = Tensor::cat(
                    &[
                        previous.shallow_clone(),
                        mean.shallow_clone(),
                        z.shallow_clone(),
                        count,
                    ],
                    -1,
                );
                let next = a.recurrent.step(&input, alignment_state.as_ref().unwrap());
                let parameters = next.value().select(0, 0).apply(&a.window);
                alignment_state = Some(next);
                if t > 0 {
                    centers = &centers
                        + parameters
                            .narrow(-1, (2 * WINDOWS) as i64, WINDOWS as i64)
                            .clamp(-8., 2.)
                            .exp()
                            .to_kind(Kind::Double);
                }
                let (p, scores) = self.window(&parameters.unsqueeze(1), &centers.unsqueeze(1), b);
                (p.squeeze_dim(1), scores.squeeze_dim(1))
            });
            let progress = learned
                .as_ref()
                .map_or(t as f64 / &expected, |(p, _)| p.shallow_clone());
            let distance = (&progress - &token_position) * &units;
            let recurrent_text = if self.local_conditioning {
                learned
                    .as_ref()
                    .map_or_else(
                        || -distance.square() * self.attention_strength,
                        |(_, scores)| scores.shallow_clone(),
                    )
                    .masked_fill(&b.text_mask.eq(0.), -1e9)
                    .softmax(-1, Kind::Float)
                    .unsqueeze(1)
                    .bmm(states)
                    .squeeze_dim(1)
            } else {
                mean.shallow_clone()
            };
            let mut input = vec![previous, recurrent_text];
            if self.global_conditioning {
                input.push(mean.shallow_clone());
            }
            input.push(z.shallow_clone());
            input.push(progress.shallow_clone());
            if self.global_conditioning {
                input.push(
                    b.text_mask
                        .sum_dim_intlist(&[1_i64][..], true, Kind::Float)
                        .log(),
                );
            }
            state = self.decoder.step(&Tensor::cat(&input, -1), &state);
            let hidden = state.value().select(0, 0);
            let logits = hidden
                .apply(&self.query)
                .unsqueeze(1)
                .bmm(&states.transpose(1, 2))
                .squeeze_dim(1)
                / (HIDDEN as f64).sqrt()
                + learned.as_ref().map_or_else(
                    || -distance.square() * self.attention_strength,
                    |(_, scores)| scores.shallow_clone(),
                );
            let attention = logits
                .masked_fill(&b.text_mask.eq(0.), -1e9)
                .softmax(-1, Kind::Float);
            let context = attention.unsqueeze(1).bmm(states).squeeze_dim(1);
            let y = Tensor::cat(&[hidden, context, z.shallow_clone()], -1).apply(&self.out);
            let component = y.narrow(-1, 3, MIXTURES as i64).argmax(-1, false);
            let parameters = y
                .narrow(-1, (3 + MIXTURES) as i64, (MIXTURES * 14) as i64)
                .reshape([batch_size, MIXTURES as i64, 14]);
            let geometry = parameters
                .gather(
                    1,
                    &component
                        .reshape([batch_size, 1, 1])
                        .expand([-1, 1, 7], true),
                    false,
                )
                .squeeze_dim(1);
            let kinds = if t == 0 {
                Tensor::zeros([batch_size], (Kind::Int64, device))
                    .one_hot(3)
                    .to_kind(Kind::Float)
            } else {
                y.narrow(-1, 0, 3)
                    .argmax(-1, false)
                    .one_hot(3)
                    .to_kind(Kind::Float)
            };
            let move_mask = kinds.narrow(-1, 0, 1);
            let curve_mask = kinds.narrow(-1, 1, 1);
            let active = &move_mask + &curve_mask;
            let geometry = Tensor::cat(
                &[
                    geometry.narrow(-1, 0, 2) * &active,
                    geometry.narrow(-1, 2, 4) * &curve_mask,
                    geometry.narrow(-1, 6, 1).abs().clamp(1e-5, 1.) * &active,
                ],
                -1,
            );
            prediction = Tensor::cat(&[geometry, kinds], -1).detach();
            output.push(y);
        }
        Tensor::stack(&output, 1)
    }
    fn loss(&self, b: &Batch, step: u64, noise: bool) -> Tensor {
        self.loss_components(b, step, noise).total()
    }
    fn loss_components(&self, b: &Batch, step: u64, noise: bool) -> LossComponents {
        let (mu, lv) = self.posterior(b);
        let (states, text_mean) = self.condition(b);
        let z = if noise && (step < CONTENT_WARMUP || step.is_multiple_of(4)) {
            Tensor::zeros_like(&mu)
        } else if noise && step % 4 == 1 {
            Tensor::randn_like(&mu)
        } else if noise {
            &mu + (&lv * 0.5).exp() * Tensor::randn_like(&mu)
        } else {
            mu.shallow_clone()
        };
        let y = if noise && rollout_probability(step) > 0. && step.is_multiple_of(2) {
            self.forward_scheduled_conditioned(b, &z, step, &states, &text_mean)
        } else {
            self.forward_conditioned(b, &z, &states, &text_mean)
        };
        let kinds = b.events.narrow(-1, 7, 3);
        let event = -(&kinds * y.narrow(-1, 0, 3).log_softmax(-1, Kind::Float)).sum_dim_intlist(
            &[-1_i64][..],
            false,
            Kind::Float,
        );
        let params = y
            .narrow(-1, (3 + MIXTURES) as i64, (MIXTURES * 14) as i64)
            .reshape([-1, b.events.size()[1], MIXTURES as i64, 14]);
        let means = params.narrow(-1, 0, 7);
        let log_sigma = params.narrow(-1, 7, 7).clamp(-9., 0.);
        let target = b.events.narrow(-1, 0, 7).unsqueeze(2);
        let move_mask = b.events.narrow(-1, 7, 1);
        let curve_mask = b.events.narrow(-1, 8, 1);
        let dims = Tensor::cat(
            &[
                &move_mask + &curve_mask,
                &move_mask + &curve_mask,
                curve_mask.repeat([1, 1, 4]),
                &move_mask + &curve_mask,
            ],
            -1,
        )
        .unsqueeze(2);
        let log_density: Tensor =
            -0.5 * ((&target - &means) / log_sigma.exp()).square() - &log_sigma - 0.918938533;
        let log_prob = (log_density * &dims).sum_dim_intlist(&[-1_i64][..], false, Kind::Float)
            + y.narrow(-1, 3, MIXTURES as i64)
                .log_softmax(-1, Kind::Float);
        let nll = -log_prob.logsumexp(-1, false);
        let active: Tensor = 1. - b.events.narrow(-1, END as i64, 1).squeeze_dim(-1);
        let reconstruction = (event * &b.event_weights).sum(Kind::Float)
            / b.event_weights.sum(Kind::Float)
            + (nll * active * 0.1 * &b.weights).sum(Kind::Float) / b.weights.sum(Kind::Float);
        let kl = (mu.square() + lv.exp() - &lv - 1.).mean(Kind::Float) * 0.5;
        let beta = (step.saturating_sub(CONTENT_WARMUP) as f64 / 2000.).min(1.) * 0.001;
        let predicted_length = text_mean.apply(&self.length).squeeze_dim(-1)
            + b.text_mask
                .sum_dim_intlist(&[1_i64][..], false, Kind::Float)
                .log();
        let target_length = b
            .mask
            .sum_dim_intlist(&[1_i64][..], false, Kind::Float)
            .log();
        let mut duration = (predicted_length - target_length)
            .square()
            .mean(Kind::Float);
        if let Some(parameters) = self.alignment_parameters(b, &z, &text_mean) {
            let delta = parameters
                .narrow(-1, (2 * WINDOWS) as i64, WINDOWS as i64)
                .clamp(-8., 2.)
                .exp();
            let centers = delta.cumsum(1, Kind::Float) - delta.narrow(1, 0, 1);
            let last = (b.mask.sum_dim_intlist(&[1_i64][..], false, Kind::Int64) - 1)
                .reshape([-1, 1, 1])
                .expand([-1, 1, WINDOWS as i64], true);
            let end = centers.gather(1, &last, false).squeeze_dim(1);
            let count = b.text_mask.sum_dim_intlist(&[1_i64][..], true, Kind::Float);
            duration += ((end - (&count - 1.)) / count).square().mean(Kind::Float);
        }
        // Relative-coordinate likelihood alone can hide accumulated layout drift.
        // Compare complete paths in source-normalized space, without flattening baselines.
        let component = y.narrow(-1, 3, MIXTURES as i64).argmax(-1, false);
        let selected = means
            .gather(
                2,
                &component
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand([-1, -1, 1, 7], true),
                false,
            )
            .squeeze_dim(2);
        let direct = local_geometry(&selected, b);
        let curvature =
            |e: &Tensor| e.narrow(-1, 0, 2) - e.narrow(-1, 2, 2) * 2. + e.narrow(-1, 4, 2);
        let curvature_error = (curvature(&selected) - curvature(&b.events)) * &b.geometry_scale;
        let curvature_loss = (curvature_error.smooth_l1_loss(
            &Tensor::zeros_like(&curvature_error),
            tch::Reduction::None,
            0.1,
        ) * &curve_mask
            * b.weights.unsqueeze(-1))
        .sum(Kind::Float)
            / (curve_mask.sum(Kind::Float) * 2.).clamp_min(1.);
        let delta = |events: &Tensor| {
            events.narrow(-1, 0, 2) * &move_mask + events.narrow(-1, 4, 2) * &curve_mask
        };
        let predicted_path = (delta(&selected) * b.mask.unsqueeze(-1)).cumsum(1, Kind::Float);
        let target_path = (delta(&b.events) * b.mask.unsqueeze(-1)).cumsum(1, Kind::Float);
        let absolute =
            trajectory_geometry(&predicted_path, &target_path, &b.mask, &b.position_scale);
        LossComponents {
            reconstruction,
            kl: kl * beta,
            duration: duration * 0.2,
            absolute: absolute * 0.2,
            geometry: direct * 5. + curvature_loss,
        }
    }
    fn export(&self, id: &str, vocab: &[String], max_events: usize) -> Result<SequenceDecoder> {
        let variables = self.store.variables();
        let linear = |weight: &str, bias: &str| -> Result<Linear> {
            let w = variables.get(weight).context("export weight")?;
            let b = variables.get(bias).context("export bias")?;
            Ok(Linear {
                weight: Vec::<f32>::try_from(w.to_device(Device::Cpu).reshape([-1]))?,
                bias: Vec::<f32>::try_from(b.to_device(Device::Cpu).reshape([-1]))?,
                input: w.size()[1] as usize,
            })
        };
        let gru = |name: &str| -> Result<Gru> {
            Ok(Gru {
                input: linear(
                    &format!("{name}.weight_ih_l0"),
                    &format!("{name}.bias_ih_l0"),
                )?,
                recurrent: linear(
                    &format!("{name}.weight_hh_l0"),
                    &format!("{name}.bias_hh_l0"),
                )?,
            })
        };
        let embed = Vec::<Vec<f32>>::try_from(self.embed.ws.to_device(Device::Cpu))?;
        let d = SequenceDecoder {
            alignment: self
                .alignment
                .as_ref()
                .map(|_| -> Result<Alignment> {
                    Ok(Alignment {
                        recurrent: gru("aligner")?,
                        window: linear("alignment_window.weight", "alignment_window.bias")?,
                    })
                })
                .transpose()?,
            em_normalization: self.alignment.is_some(),
            style_domain: "personal".into(),
            whole_run: true,
            length_per_token: true,
            canonical_feedback: true,
            attention_strength: self.attention_strength as f32,
            token_space_attention: self.token_space_attention,
            global_conditioning: self.global_conditioning,
            local_conditioning: self.local_conditioning,
            glyph_head: None,
            schema_version: 3,
            kind: "sequence-cvae-v3".into(),
            model_id: id.into(),
            vocabulary: vocab.to_vec(),
            embedding: embed,
            encoder: gru("encoder")?,
            decoder: gru("decoder")?,
            query: linear("query.weight", "query.bias")?,
            output: linear("output.weight", "output.bias")?,
            max_events,
            metrics: BTreeMap::new(),
            length: linear("length.weight", "length.bias")?,
        };
        d.validate()?;
        Ok(d)
    }
}
pub(super) struct Adam {
    states: BTreeMap<String, (Tensor, Tensor)>,
    pub(super) step: u64,
}
impl Adam {
    pub(super) fn new() -> Self {
        Self {
            states: BTreeMap::new(),
            step: 0,
        }
    }
    pub(super) fn fork(&self) -> Self {
        Self {
            step: self.step,
            states: self
                .states
                .iter()
                .map(|(n, (m, v))| (n.clone(), (m.shallow_clone(), v.shallow_clone())))
                .collect(),
        }
    }
    pub(super) fn update(&mut self, store: &nn::VarStore) -> Result<()> {
        self.step += 1;
        tch::no_grad(|| -> Result<()> {
            let variables: BTreeMap<_, _> = store.variables().into_iter().collect();
            let norms: Vec<_> = variables
                .values()
                .map(|w| w.grad().square().sum(Kind::Float))
                .collect();
            let norm = Tensor::stack(&norms, 0)
                .sum(Kind::Float)
                .sqrt()
                .double_value(&[]);
            ensure!(norm.is_finite(), "nonfinite gradient");
            let clip = (1. / (norm + 1e-8)).min(1.);
            for (name, mut w) in variables {
                let grad = w.grad() * clip;
                ensure!(grad.device() == w.device(), "gradient off device");
                let (m, v) = self
                    .states
                    .entry(name)
                    .or_insert_with(|| (Tensor::zeros_like(&w), Tensor::zeros_like(&w)));
                *m = &*m * 0.9 + &grad * 0.1;
                *v = &*v * 0.999 + grad.square() * 0.001;
                let delta = (&*m / (1. - 0.9_f64.powf(self.step as f64)))
                    / ((&*v / (1. - 0.999_f64.powf(self.step as f64))).sqrt() + 1e-8)
                    * 0.0003;
                let _ = w.f_sub_(&delta)?;
                w.zero_grad();
            }
            Ok(())
        })
    }
    pub(super) fn save(&self, path: &Path) -> Result<()> {
        let tensors: Vec<_> = self
            .states
            .iter()
            .flat_map(|(name, (m, v))| {
                [
                    (format!("m.{name}"), m.to_device(Device::Cpu)),
                    (format!("v.{name}"), v.to_device(Device::Cpu)),
                ]
            })
            .collect();
        Tensor::write_safetensors(&tensors, path)?;
        Ok(())
    }
    pub(super) fn restore(path: &Path, step: u64, store: &nn::VarStore) -> Result<Self> {
        let mut tensors: BTreeMap<_, _> = Tensor::read_safetensors(path)?.into_iter().collect();
        let mut states = BTreeMap::new();
        for (name, w) in store.variables() {
            let m = tensors
                .remove(&format!("m.{name}"))
                .context("missing first moment")?;
            let v = tensors
                .remove(&format!("v.{name}"))
                .context("missing second moment")?;
            ensure!(
                m.size() == w.size() && v.size() == w.size(),
                "optimizer shape mismatch"
            );
            states.insert(name, (m.to_device(w.device()), v.to_device(w.device())));
        }
        ensure!(tensors.is_empty(), "extra optimizer states");
        Ok(Self { states, step })
    }
}
struct Prepared {
    sample: Sample,
    events: Vec<[f32; FEATURES]>,
    tokens: Vec<i64>,
    weights: Vec<f32>,
    event_weights: Vec<f32>,
    geometry_scale: f32,
    position_scale: f32,
}
fn prepare(samples: Vec<Sample>, vocab: &[String]) -> Result<Vec<Prepared>> {
    prepare_using(samples, vocab, encode_em)
}
fn encode_em(sample: &Sample) -> Result<sequence::Sequence> {
    sequence::encode_em(sample, ink_layout::tokenize(&sample.label)?.len())
}
fn prepare_using(
    samples: Vec<Sample>,
    vocab: &[String],
    encode: fn(&Sample) -> Result<sequence::Sequence>,
) -> Result<Vec<Prepared>> {
    samples
        .into_iter()
        .map(|sample| {
            let encoded = encode(&sample)?;
            let bounds = ink_corpus::bounds(&sample);
            let height = (bounds[3] - bounds[1])
                .max(sample.strokes.iter().map(|s| s.width).fold(0_f32, f32::max))
                .max(1e-6);
            let height_gain = (encoded.scale / height).clamp(1., 32.);
            let events = encoded.events;
            let weights = sequence::training_weights(&events)?;
            let event_weights = sequence::event_weights(&events)?;
            let tokens: Vec<i64> = ink_layout::tokenize(&sample.label)?
                .iter()
                .map(|c| {
                    vocab
                        .binary_search(c)
                        .map(|i| i as i64)
                        .map_err(|_| anyhow::anyhow!("{}: untrained token {c:?}", sample.sample_id))
                })
                .collect::<Result<_>>()?;
            let geometry_scale =
                height_gain * (events.len() as f32 / tokens.len().max(1) as f32).clamp(1., 64.);
            Ok(Prepared {
                sample,
                events,
                tokens,
                weights,
                event_weights,
                geometry_scale,
                position_scale: height_gain,
            })
        })
        .collect()
}
fn batch(rows: &[&Prepared], device: Device, augment: bool, step: u64) -> Batch {
    let n = rows.len();
    let t = rows.iter().map(|r| r.events.len()).max().unwrap();
    let l = rows.iter().map(|r| r.tokens.len()).max().unwrap();
    let mut tokens = vec![0_i64; n * l];
    let mut text_mask = vec![0_f32; n * l];
    let mut events = vec![0_f32; n * t * FEATURES];
    let mut mask = vec![0_f32; n * t];
    let mut weights = vec![0_f32; n * t];
    let mut event_weights = vec![0_f32; n * t];
    let geometry_scale: Vec<_> = rows.iter().map(|r| r.geometry_scale).collect();
    let position_scale: Vec<_> = rows.iter().map(|r| r.position_scale).collect();
    let mut previous = vec![0_f32; n * t * FEATURES];
    for (i, row) in rows.iter().enumerate() {
        weights[i * t..i * t + row.weights.len()].copy_from_slice(&row.weights);
        event_weights[i * t..i * t + row.event_weights.len()].copy_from_slice(&row.event_weights);
        let (sx, sy) = if augment {
            let noise = ink_neural::normal(step.wrapping_mul(7919) + i as u64, 2);
            (1. + noise[0] * 0.025, 1. + noise[1] * 0.025)
        } else {
            (1., 1.)
        };
        for (j, token) in row.tokens.iter().enumerate() {
            tokens[i * l + j] = *token;
            text_mask[i * l + j] = 1.;
        }
        for (j, event) in row.events.iter().enumerate() {
            let mut e = *event;
            for k in 0..3 {
                e[k * 2] *= sx;
                e[k * 2 + 1] *= sy;
            }
            events[(i * t + j) * FEATURES..(i * t + j + 1) * FEATURES].copy_from_slice(&e);
            mask[i * t + j] = 1.;
            if j + 1 < t {
                previous[(i * t + j + 1) * FEATURES..(i * t + j + 2) * FEATURES]
                    .copy_from_slice(&e);
            }
        }
    }
    let tensor = |x: &[f32], shape: &[i64]| Tensor::from_slice(x).reshape(shape).to_device(device);
    Batch {
        tokens: Tensor::from_slice(&tokens)
            .reshape([n as i64, l as i64])
            .to_device(device),
        text_mask: tensor(&text_mask, &[n as i64, l as i64]),
        events: tensor(&events, &[n as i64, t as i64, FEATURES as i64]),
        mask: tensor(&mask, &[n as i64, t as i64]),
        weights: tensor(&weights, &[n as i64, t as i64]),
        previous: tensor(&previous, &[n as i64, t as i64, FEATURES as i64]),
        event_weights: tensor(&event_weights, &[n as i64, t as i64]),
        geometry_scale: tensor(&geometry_scale, &[n as i64, 1, 1]),
        position_scale: tensor(&position_scale, &[n as i64, 1, 1]),
    }
}
fn parity(net: &Network, d: &SequenceDecoder, rows: &[Prepared], device: Device) -> Result<f64> {
    let row = rows.first().context("empty parity input")?;
    let b = batch(&[row], device, false, 0);
    let z = Tensor::zeros([1, STYLE as i64], (Kind::Float, device));
    let y = tch::no_grad(|| net.forward(&b, &z));
    let (native_states, native_mean) = tch::no_grad(|| net.condition(&b));
    let _ = (native_states, native_mean);
    let (states, mean) = d.condition(&row.sample.label)?;
    let mut h = vec![0.; HIDDEN];
    let mut previous = [0.; FEATURES];
    let mut error = 0_f64;
    for (i, e) in row.events.iter().enumerate() {
        let (next, output) = d.step(
            &previous,
            &h,
            &states,
            &mean,
            &[0.; STYLE],
            i as f32 / row.events.len() as f32,
        )?;
        h = next;
        let target = Vec::<f32>::try_from(y.get(0).get(i as i64).to_device(Device::Cpu))?;
        for (a, b) in output.iter().zip(target) {
            error = error.max((a - b).abs() as f64);
        }
        previous = *e;
    }
    ensure!(error < 0.0002, "sequence Rust decoder parity error {error}");
    Ok(error)
}
fn allocate(models: &Path) -> Result<std::path::PathBuf> {
    fs::create_dir_all(models)?;
    for i in 0..100000 {
        let path = models.join(format!("m{i:03}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("model IDs exhausted")
}
/// Explicit transfer changes the corpus/recipe and resets optimizers. Resume is
/// still exact. Tokens move by label, never by a coincidentally matching index.
fn transfer_parent(
    net: &Network,
    parent: &Path,
    vocab: &[String],
    samples: &[Sample],
) -> Result<Value> {
    let manifest = super::show(parent)?;
    ensure!(
        manifest["kind"] == "sequence-cvae-v3",
        "parent must be a compatible sequence network"
    );
    ensure!(
        manifest["checkpoint_sha256"] == file_hash(&parent.join("checkpoint.safetensors"))?,
        "parent checkpoint checksum mismatch"
    );
    ensure!(
        manifest["train_lineage_sha256"] == file_hash(&parent.join("train-lineage.json"))?,
        "parent lineage checksum mismatch"
    );
    let lineage: Vec<Value> =
        serde_json::from_slice(&fs::read(parent.join("train-lineage.json"))?)?;
    let heldout: BTreeSet<_> = samples
        .iter()
        .filter(|s| s.split != "train")
        .map(|s| s.source_sha256.as_str())
        .collect();
    ensure!(
        !lineage.iter().any(|row| row["source_sha256"]
            .as_str()
            .is_some_and(|s| heldout.contains(s))),
        "parent training sources overlap current holdout"
    );
    let old_vocab: Vec<String> =
        serde_json::from_slice(&fs::read(parent.join("vocabulary.json"))?)?;
    let old: BTreeMap<_, _> = Tensor::read_safetensors(parent.join("checkpoint.safetensors"))?
        .into_iter()
        .collect();
    let mut copied = 0;
    tch::no_grad(|| -> Result<()> {
        for (name, mut weight) in net.store.variables() {
            if name.starts_with("length.")
                && manifest["recipe"]["duration_parameterization"]
                    != serde_json::from_str::<Value>(RECIPE)?["duration_parameterization"]
            {
                continue;
            }
            if (name.starts_with("aligner.") || name.starts_with("alignment_window."))
                && !old.contains_key(&name)
            {
                continue;
            }
            let source = old
                .get(&name)
                .with_context(|| format!("parent missing tensor {name}"))?
                .to_device(net.store.device());
            if name == "embed.weight" {
                ensure!(
                    source.size() == [old_vocab.len() as i64, EMBED as i64],
                    "parent embedding dimensions"
                );
                for (i, token) in vocab.iter().enumerate() {
                    if let Ok(j) = old_vocab.binary_search(token) {
                        weight.get(i as i64).f_copy_(&source.get(j as i64))?;
                        copied += 1;
                    }
                }
            } else if name == "decoder.weight_ih_l0"
                && source.size() == [3 * HIDDEN as i64, decoder_input_width(false) as i64]
                && weight.size() == [3 * HIDDEN as i64, decoder_input_width(true) as i64]
            {
                // Preserve old previous/local/style/progress columns exactly.
                // New global-context/count columns begin at zero; fresh Adam
                // learns them without reinterpreting any historical weights.
                let _ = weight.zero_();
                let prefix = (FEATURES + HIDDEN) as i64;
                weight
                    .narrow(1, 0, prefix)
                    .f_copy_(&source.narrow(1, 0, prefix))?;
                weight
                    .narrow(1, prefix + HIDDEN as i64, (STYLE + 1) as i64)
                    .f_copy_(&source.narrow(1, prefix, (STYLE + 1) as i64))?;
            } else {
                ensure!(
                    source.size() == weight.size(),
                    "parent tensor shape mismatch: {name}"
                );
                weight.f_copy_(&source)?;
            }
        }
        Ok(())
    })?;
    Ok(
        json!({"parent":parent,"checkpoint_sha256":manifest["checkpoint_sha256"],"parent_dataset_sha256":manifest["dataset_sha256"],"parent_recipe":manifest["recipe"],"parent_training_domain":manifest["training_domain"].as_str().unwrap_or("personal"),"public_pretraining":manifest["training_domain"]=="public"||manifest["public_pretraining"]==true,"transfer":"exact token mapping; fresh optimizers; current source holdout checked","copied_tokens":copied,"new_tokens":vocab.len()-copied}),
    )
}
pub fn train(options: super::TrainOptions<'_>) -> Result<Value> {
    let super::TrainOptions {
        model,
        corpus,
        models,
        seconds,
        resume,
        repo,
        cpu,
        gpu_index,
    } = options;
    ensure!(
        (1..=28800).contains(&seconds),
        "time budget must be 1..28800 seconds"
    );
    ensure!(!resume || model.is_some(), "resume requires --model");
    let preparation = super::progress::Progress::spinner("Prepare model and corpus");
    let samples = load(corpus)?;
    audit(&samples)?;
    let training_domain = super::style::domain(&samples)?;
    let train: Vec<_> = samples
        .iter()
        .filter(|r| r.split == "train")
        .cloned()
        .collect();
    ensure!(!train.is_empty(), "empty train split");
    let mut tokens: BTreeSet<String> = BTreeSet::new();
    for r in &train {
        tokens.extend(
            ink_layout::tokenize(&r.label)
                .with_context(|| format!("training grammar for {}", r.sample_id))?,
        );
    }
    tokens.insert("\0".into());
    let vocab: Vec<_> = tokens.into_iter().collect();
    let rows = prepare(train, &vocab)?;
    let max_events = rows.iter().map(|r| r.events.len()).max().unwrap() * 2 + 32;
    let (device, hardware) = if cpu {
        (Device::Cpu, json!({"device":"CPU"}))
    } else {
        super::rocm::device(gpu_index)?
    };
    tch::set_num_threads(2);
    tch::manual_seed(419);
    let mut net = Network::new(device, vocab.len());
    let mut adam = Adam::new();
    let hash = file_hash(corpus)?;
    let mut previous_seconds = 0.;
    let mut execution = super::execution::ExecutionPlan::new(cpu);
    let mut parent_model = None;
    let target = if resume {
        let p = super::resolve(model, models)?;
        let m = super::show(&p)?;
        execution = super::execution::ExecutionPlan::resume(&m)?;
        ensure!(
            m["kind"] == "sequence-cvae-v3"
                && m["recipe"] == serde_json::from_str::<Value>(RECIPE)?
                && m["dataset_sha256"] == hash,
            "resume recipe or corpus mismatch"
        );
        if p.join("glyph-training.json").exists() {
            let current: Value = serde_json::from_str(include_str!(
                "../../../recipes/compositional-vector-v10.json"
            ))?;
            let state_path = p.join("glyph-training.json");
            let mut state: Value = serde_json::from_slice(&fs::read(&state_path)?)?;
            if state.get("recipe").is_none() {
                ensure!(
                    m["glyph_recipe"] == current,
                    "glyph recipe metadata missing or incompatible"
                );
                state["recipe"] = current;
                save(&state_path, &state)?;
            }
        }
        let old: Vec<String> = serde_json::from_slice(&fs::read(p.join("vocabulary.json"))?)?;
        ensure!(old == vocab, "resume vocabulary mismatch");
        for key in ["checkpoint", "optimizer"] {
            ensure!(
                m[format!("{key}_sha256")] == file_hash(&p.join(format!("{key}.safetensors")))?,
                "resume integrity mismatch"
            );
        }
        net.store.load(p.join("checkpoint.safetensors"))?;
        adam = Adam::restore(
            &p.join("optimizer.safetensors"),
            m["updates"].as_u64().context("updates")?,
            &net.store,
        )?;
        previous_seconds = m["training_seconds"].as_f64().context("training seconds")?;
        if m.get("glyph_training").is_none() && p.join("glyph-training.json").exists() {
            let phase: Value = serde_json::from_slice(&fs::read(p.join("glyph-training.json"))?)?;
            previous_seconds += phase["seconds"].as_f64().unwrap_or(0.);
        }
        parent_model = m["parent_model"].as_str().map(std::path::PathBuf::from);
        p
    } else {
        let mut parents = vec![];
        if models.exists() {
            for entry in fs::read_dir(models)? {
                let p = entry?.path();
                if p.join("checkpoint.safetensors").exists()
                    && let Ok(m) = super::show(&p)
                    && m["kind"] == "sequence-cvae-v3"
                    && m["recipe"] == serde_json::from_str::<Value>(RECIPE)?
                    && m["dataset_sha256"] == hash
                    && serde_json::from_slice::<Vec<String>>(&fs::read(p.join("vocabulary.json"))?)?
                        == vocab
                {
                    parents.push(p);
                }
            }
        }
        parents.sort();
        if let Some(requested) = model {
            let selected = super::resolve(Some(requested), models)?;
            parents = vec![selected];
        }
        let mut transfer = None;
        if let Some(p) = parents.last() {
            transfer = Some(transfer_parent(&net, p, &vocab, &samples)?);
            parent_model = Some(p.clone());
        }
        let allocated = allocate(models)?;
        if let Some(mut transfer) = transfer {
            transfer["dataset_sha256"] = json!(hash);
            save(&allocated.join("parent-model.json"), &transfer)?;
        }
        allocated
    };
    ensure!(
        previous_seconds < 28800.,
        "eight-hour model budget exhausted"
    );
    save(&target.join("corpus-audit.json"), &audit(&samples)?)?;
    let total_budget = (seconds as f64).min(28800. - previous_seconds);
    let budget = total_budget;
    let id = target
        .file_name()
        .context("model ID")?
        .to_string_lossy()
        .to_string();
    let run = repo.join("runs").join(&id);
    fs::create_dir_all(&run)?;
    let groups = execution.groups(&rows.iter().map(|r| r.events.len()).collect::<Vec<_>>());
    // Small immutable training corpora fit comfortably on the GPU. Cache padded
    // inputs, never activations; for larger corpora retain on-demand transfers.
    let cache_bytes: usize = groups
        .iter()
        .map(|indices| {
            let events = indices.iter().map(|&i| rows[i].events.len()).max().unwrap();
            let tokens = indices.iter().map(|&i| rows[i].tokens.len()).max().unwrap();
            indices.len() * (events * (2 * FEATURES + 3) * 4 + tokens * 12 + 4)
        })
        .sum();
    let cached = if device.is_cuda() && cache_bytes <= 256 * 1024 * 1024 {
        Some(
            groups
                .iter()
                .map(|indices| {
                    batch(
                        &indices.iter().map(|&i| &rows[i]).collect::<Vec<_>>(),
                        device,
                        false,
                        0,
                    )
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let started = Instant::now();
    let mut saved = Instant::now();
    let initial = tch::no_grad(|| {
        net.loss(&batch(&[&rows[0]], device, false, 0), adam.step, false)
            .double_value(&[])
    });
    let mut final_loss = initial;
    let mut trace = vec![];
    let mut seen = BTreeSet::new();
    super::progress::log(format!(
        "Training {id}: {} whole train samples, {} tokens, {:?}; {:.0}s budget; up to {} rows/batch, {} batches/epoch; GPU input cache {}",
        rows.len(),
        vocab.len(),
        device,
        budget,
        execution.max_batch_samples,
        groups.len(),
        if cached.is_some() {
            format!("{:.1} MiB", cache_bytes as f64 / 1_048_576.)
        } else {
            "off".into()
        }
    ));
    preparation.finish(format!("{id} · {} train rows", rows.len()));
    let mut progress =
        super::progress::EpochProgress::new(&format!("{id} · sequence"), groups.len(), adam.step);
    while started.elapsed().as_secs_f64() < budget {
        progress.start();
        let epoch = adam.step / groups.len() as u64;
        let slot = adam.step as usize % groups.len();
        let mut order: Vec<_> = (0..groups.len()).collect();
        let mut rng = epoch.wrapping_add(419);
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            order.swap(i, (rng as usize) % (i + 1));
        }
        let selected = &groups[order[slot]];
        let chosen: Vec<_> = selected.iter().map(|i| &rows[*i]).collect();
        for i in selected {
            seen.insert(*i);
        }
        tch::manual_seed((419 + adam.step) as i64);
        let prepared;
        let b = if let Some(cached) = &cached {
            &cached[order[slot]]
        } else {
            prepared = batch(&chosen, device, false, adam.step);
            &prepared
        };
        let loss = net.loss(b, adam.step, true);
        ensure!(loss.device() == device, "loss off selected device");
        final_loss = loss.double_value(&[]);
        ensure!(final_loss.is_finite(), "nonfinite sequence loss");
        loss.backward();
        adam.update(&net.store)?;
        if let Some(summary) = progress.update(final_loss, chosen.len()) {
            trace.push(json!({"epoch":summary.epoch,"update":adam.step,"mean_loss":summary.mean_loss,"seconds":started.elapsed().as_secs_f64()}));
        }
        if saved.elapsed().as_secs() >= 60 {
            progress.message("Saving checkpoint and decoder…");
            publish(
                &net,
                &adam,
                &rows,
                &vocab,
                max_events,
                &target,
                &id,
                &hash,
                &hardware,
                previous_seconds + started.elapsed().as_secs_f64(),
                initial,
                final_loss,
                device,
                repo,
                &seen,
                &execution,
                groups.len(),
            )?;
            save(&run.join("loss.json"), &trace)?;
            saved = Instant::now();
        }
    }
    progress.message("Saving final sequence checkpoint…");
    let mut result = publish(
        &net,
        &adam,
        &rows,
        &vocab,
        max_events,
        &target,
        &id,
        &hash,
        &hardware,
        previous_seconds + started.elapsed().as_secs_f64(),
        initial,
        final_loss,
        device,
        repo,
        &seen,
        &execution,
        groups.len(),
    )?;
    save(&run.join("loss.json"), &trace)?;
    progress.finish();
    let head_seconds = if training_domain == "public" {
        0.
    } else {
        total_budget
            * serde_json::from_str::<Value>(RECIPE)?["head_training_fraction"]
                .as_f64()
                .unwrap_or(0.)
    };
    let head = if head_seconds >= 1. {
        super::compositional::train(corpus, &target, device, head_seconds, resume)?
    } else {
        json!({"status":"skipped","reason":"current recipe dedicates its budget to complete-trajectory content and geometry"})
    };
    let mut decoder = SequenceDecoder::load(&target.join("decoder.json"))?;
    decoder.glyph_head = if target.join("glyph-decoder.json").exists() {
        Some(serde_json::from_slice(&fs::read(
            target.join("glyph-decoder.json"),
        )?)?)
    } else {
        None
    };
    decoder.validate()?;
    save(&target.join("decoder.json"), &decoder)?;
    let mut manifest: Value = serde_json::from_slice(&fs::read(target.join("manifest.json"))?)?;
    manifest["decoder_sha256"] = json!(file_hash(&target.join("decoder.json"))?);
    manifest["glyph_recipe"] = serde_json::from_str(super::compositional::RECIPE)?;
    manifest["parent_model"] = json!(parent_model);
    manifest["glyph_conditions"] =
        json!(decoder.glyph_head.as_ref().map_or(0, |g| g.supported.len()));
    manifest["glyph_training"] = head.clone();
    if decoder.glyph_head.is_some() {
        manifest["glyph_lineage_sha256"] = json!(file_hash(&target.join("glyph-lineage.json"))?);
        manifest["glyph_decoder_sha256"] = json!(file_hash(&target.join("glyph-decoder.json"))?);
    }
    manifest["training_seconds"] = json!(previous_seconds + started.elapsed().as_secs_f64());
    save(&target.join("manifest.json"), &manifest)?;
    result["glyph_conditions"] = manifest["glyph_conditions"].clone();
    result["compositional"] = head;
    Ok(result)
}
fn resume_contract(
    net: &Network,
    adam: &Adam,
    rows: &[Prepared],
    vocab: &[String],
    target: &Path,
    device: Device,
) -> Result<f64> {
    let mut uninterrupted = Network::with_architecture(
        device,
        vocab.len(),
        net.global_conditioning,
        net.alignment.is_some(),
    );
    uninterrupted.store.copy(&net.store)?;
    let mut restored = Network::with_architecture(
        device,
        vocab.len(),
        net.global_conditioning,
        net.alignment.is_some(),
    );
    restored.store.load(target.join("checkpoint.safetensors"))?;
    let mut a = adam.fork();
    let mut b = Adam::restore(
        &target.join("optimizer.safetensors"),
        adam.step,
        &restored.store,
    )?;
    let batch = batch(&[&rows[0]], device, false, adam.step);
    let x = uninterrupted.loss(&batch, a.step, false);
    let y = restored.loss(&batch, b.step, false);
    ensure!(
        (x.double_value(&[]) - y.double_value(&[])).abs() < 1e-6,
        "checkpoint reload changed reconstruction"
    );
    x.backward();
    y.backward();
    a.update(&uninterrupted.store)?;
    b.update(&restored.store)?;
    let other = restored.store.variables();
    let mut error = 0_f64;
    for (name, weight) in uninterrupted.store.variables() {
        let expected = other.get(&name).context("restored parameter")?;
        error = error.max((weight - expected).abs().max().double_value(&[]));
    }
    ensure!(error < 1e-6, "restored actual Adam update differs: {error}");
    Ok(error)
}

#[allow(clippy::too_many_arguments)]
fn publish(
    net: &Network,
    adam: &Adam,
    rows: &[Prepared],
    vocab: &[String],
    max_events: usize,
    target: &Path,
    id: &str,
    hash: &str,
    hardware: &Value,
    seconds: f64,
    initial: f64,
    loss: f64,
    device: Device,
    repo: &Path,
    seen: &BTreeSet<usize>,
    execution: &super::execution::ExecutionPlan,
    batches_per_epoch: usize,
) -> Result<Value> {
    net.store.save(target.join("checkpoint.safetensors"))?;
    adam.save(&target.join("optimizer.safetensors"))?;
    let resume_error = resume_contract(net, adam, rows, vocab, target, device)?;
    let mut decoder = net.export(id, vocab, max_events)?;
    decoder.style_domain =
        super::style::domain(&rows.iter().map(|r| r.sample.clone()).collect::<Vec<_>>())?.into();
    if target.join("glyph-decoder.json").exists() {
        decoder.glyph_head = Some(serde_json::from_slice(&fs::read(
            target.join("glyph-decoder.json"),
        )?)?);
    }
    let error = parity(net, &decoder, rows, device)?;
    save(&target.join("decoder.json"), &decoder)?;
    save(&target.join("vocabulary.json"), &vocab)?;
    let lineage:Vec<_>=rows.iter().enumerate().map(|(i,r)|json!({"sample_id":r.sample.sample_id,"source_sha256":r.sample.source_sha256,"split":"train","events":r.events.len(),"visited_this_run":seen.contains(&i),"use":"whole-sample supervision; no character segmentation"})).collect();
    save(&target.join("train-lineage.json"), &lineage)?;
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()?;
    let diff = std::process::Command::new("git")
        .args(["diff", "HEAD"])
        .current_dir(repo)
        .output()?;
    let batches_per_epoch = batches_per_epoch as u64;
    let manifest = json!({"schema_version":3,"id":id,"kind":"sequence-cvae-v3","approved_for_default":false,"recipe":serde_json::from_str::<Value>(RECIPE)?,"dataset_sha256":hash,"source_commit":String::from_utf8_lossy(&git.stdout).trim(),"source_diff_sha256":ink_corpus::hash(&diff.stdout),"decoder_sha256":file_hash(&target.join("decoder.json"))?,"vocabulary_sha256":file_hash(&target.join("vocabulary.json"))?,"checkpoint_sha256":file_hash(&target.join("checkpoint.safetensors"))?,"optimizer_sha256":file_hash(&target.join("optimizer.safetensors"))?,"train_lineage_sha256":file_hash(&target.join("train-lineage.json"))?,"nix_lock_sha256":file_hash(&repo.join("flake.lock"))?,"updates":adam.step,"epochs":adam.step / batches_per_epoch,"epoch_batches_completed":adam.step % batches_per_epoch,"batches_per_epoch":batches_per_epoch,"training_samples":rows.len(),"visited_this_run":seen.len(),"input_tokens":vocab.len(),"training_seconds":seconds,"hardware":hardware,"initial_loss":initial,"final_loss":loss,"rust_decoder_max_error":error,"adam_resume_max_error":resume_error,"limitations":["sequence content and style require generation review","supplied labels lack independent second review","no inferred character alignment"]});
    let mut manifest = manifest;
    manifest["execution_plan"] = json!(execution);
    manifest["training_domain"] = json!(decoder.style_domain);
    manifest["rendering"] =
        json!("joint complete sequence; all written marks neural; no artificial noise or joins");
    if target.join("parent-model.json").exists() {
        let parent: Value = serde_json::from_slice(&fs::read(target.join("parent-model.json"))?)?;
        manifest["public_pretraining"] = parent["public_pretraining"].clone();
        manifest["parent_checkpoint_sha256"] = parent["checkpoint_sha256"].clone();
    }
    save(&target.join("manifest.json"), &manifest)?;
    Ok(
        json!({"model":target,"updates":adam.step,"training_samples":rows.len(),"glyph_labels":vocab.len()-1,"initial_loss":initial,"final_loss":loss,"approved_for_default":false,"rust_decoder_max_error":error}),
    )
}

pub fn evaluate(model: &Path, corpus: &Path, split: &str, out: &Path) -> Result<Value> {
    ensure!(
        ["train", "validation", "test"].contains(&split),
        "unknown split"
    );
    let manifest = super::show(model)?;
    ensure!(
        manifest["dataset_sha256"] == file_hash(corpus)?,
        "evaluation corpus mismatch"
    );
    let d = SequenceDecoder::load(&model.join("decoder.json"))?;
    let all = load(corpus)?;
    audit(&all)?;
    let requested: Vec<_> = all.into_iter().filter(|r| r.split == split).collect();
    ensure!(!requested.is_empty(), "empty evaluation split");
    let mut failures = vec![];
    let mut accepted = vec![];
    for s in &requested {
        let encode = if d.em_normalization {
            encode_em
        } else if manifest["recipe"].get("normalization").is_some() {
            sequence::encode
        } else {
            sequence::encode_legacy_control_bounds
        };
        match prepare_using(vec![s.clone()], &d.vocabulary, encode) {
            Ok(mut p) => accepted.push(p.remove(0)),
            Err(e) => failures.push(json!({"sample_id":s.sample_id,"error":e.to_string()})),
        }
    }
    tch::set_num_threads(2);
    let mut net = Network::with_architecture(
        Device::Cpu,
        d.vocabulary.len(),
        d.global_conditioning,
        d.alignment.is_some(),
    );
    net.attention_strength = d.attention_strength as f64;
    net.local_conditioning = d.local_conditioning;
    net.token_space_attention = d.token_space_attention;
    net.store.load(model.join("checkpoint.safetensors"))?;
    let mut posterior_sse = 0.;
    let mut prior_sse = 0.;
    let mut coordinates = 0.;
    let mut correct = 0.;
    let mut events = 0.;
    let mut latent_means = vec![];
    let mut kl = 0.;
    let mut durations = 0.;
    let progress = super::progress::Progress::count(&format!("Evaluate {split}"), accepted.len());
    let mut evaluated = 0;
    accepted.sort_by_key(|r| r.events.len());
    for chunk in accepted.chunks(8) {
        let selected: Vec<_> = chunk.iter().collect();
        let b = batch(&selected, Device::Cpu, false, 0);
        let (mu, lv) = tch::no_grad(|| net.posterior(&b));
        let y = tch::no_grad(|| net.forward(&b, &mu));
        let prior = tch::no_grad(|| net.forward(&b, &Tensor::zeros_like(&mu)));
        let coordinate_mask = Tensor::cat(
            &[
                b.events.narrow(-1, 7, 1) + b.events.narrow(-1, 8, 1),
                b.events.narrow(-1, 7, 1) + b.events.narrow(-1, 8, 1),
                b.events.narrow(-1, 8, 1).repeat([1, 1, 4]),
            ],
            -1,
        ) * b.mask.unsqueeze(-1);
        let target = b.events.narrow(-1, 0, 6);
        let prediction = |output: &Tensor| {
            let params = output
                .narrow(-1, (3 + MIXTURES) as i64, (MIXTURES * 14) as i64)
                .reshape([-1, b.events.size()[1], MIXTURES as i64, 14]);
            let weights = output
                .narrow(-1, 3, MIXTURES as i64)
                .softmax(-1, Kind::Float)
                .unsqueeze(-1);
            (params.narrow(-1, 0, 6) * weights).sum_dim_intlist(&[2_i64][..], false, Kind::Float)
        };
        posterior_sse += ((prediction(&y) - &target).square() * &coordinate_mask)
            .sum(Kind::Float)
            .double_value(&[]);
        prior_sse += ((prediction(&prior) - &target).square() * &coordinate_mask)
            .sum(Kind::Float)
            .double_value(&[]);
        coordinates += coordinate_mask.sum(Kind::Float).double_value(&[]);
        correct += (y
            .narrow(-1, 0, 3)
            .argmax(-1, false)
            .eq_tensor(&b.events.narrow(-1, 7, 3).argmax(-1, false))
            .to_kind(Kind::Float)
            * &b.mask)
            .sum(Kind::Float)
            .double_value(&[]);
        events += b.mask.sum(Kind::Float).double_value(&[]);
        kl += (mu.square() + lv.exp() - &lv - 1.)
            .sum(Kind::Float)
            .double_value(&[])
            * 0.5;
        latent_means.extend(Vec::<Vec<f32>>::try_from(mu)?);
        let (_, mean) = tch::no_grad(|| net.condition(&b));
        let log_length = mean.apply(&net.length).squeeze_dim(-1);
        let predicted = if d.length_per_token {
            log_length
                + b.text_mask
                    .sum_dim_intlist(&[1_i64][..], false, Kind::Float)
                    .log()
        } else {
            log_length
        };
        let predicted = predicted.exp();
        let actual = b.mask.sum_dim_intlist(&[1_i64][..], false, Kind::Float);
        durations += ((predicted - &actual).abs() / actual)
            .sum(Kind::Float)
            .double_value(&[]);
        evaluated += chunk.len();
        progress.set(evaluated);
    }
    progress.set(accepted.len());
    progress.finish(format!(
        "{} samples · {} failures",
        accepted.len(),
        failures.len()
    ));
    let n = accepted.len();
    let variance: Vec<_> = (0..STYLE)
        .map(|j| {
            if n == 0 {
                return 0.;
            }
            let mean = latent_means.iter().map(|m| m[j] as f64).sum::<f64>() / n as f64;
            latent_means
                .iter()
                .map(|m| (m[j] as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64
        })
        .collect();
    let ratio = |a: f64, b: f64| if b > 0. { Some(a / b) } else { None };
    let probes = raw_sequence_probes(
        &d,
        out.parent().unwrap_or(Path::new(".")),
        manifest["decoder_sha256"]
            .as_str()
            .context("decoder checksum")?,
    )?;
    let result = json!({"free_sequence_examples":probes,"model":manifest["id"],"split":split,"whole_samples":requested.len(),"evaluated_samples":n,"failures":failures,"geometry":{"teacher_forced_posterior_delta_rmse":ratio(posterior_sse,coordinates).map(f64::sqrt),"teacher_forced_prior_delta_rmse":ratio(prior_sse,coordinates).map(f64::sqrt),"event_accuracy":ratio(correct,events),"duration_relative_error":ratio(durations,n as f64)},"latent":{"posterior_mean_variance":variance,"mean_kl":ratio(kl,(n*STYLE)as f64)},"content":"requires free-generation recognition/visual review; teacher forcing is not content accuracy","style":"not independently scored"});
    save(out, &result)?;
    Ok(result)
}
fn raw_sequence_probes(d: &SequenceDecoder, parent: &Path, checksum: &str) -> Result<Vec<Value>> {
    let mut decoder = d.clone();
    decoder.glyph_head = None;
    let folder = parent.join(format!("sequence-probes-{}", d.model_id));
    fs::create_dir_all(&folder)?;
    let mut probes = vec![];
    for (i, input) in ["f", "k", "H", "0", "Hello"].iter().enumerate() {
        let generated = (|| -> Result<Value> {
            let strokes = decoder.generate(input, 419, 0., 419)?;
            let sample = Sample {
                sample_id: "generated-probe".into(),
                label: input.to_string(),
                source_sha256: String::new(),
                split: "generated".into(),
                strokes: strokes.clone(),
                metadata: Value::Null,
            };
            let b = ink_corpus::bounds(&sample);
            let scale = 1. / (b[3] - b[1]).max(0.05);
            let drawing = ink_layout::Drawing {
                schema_version: 3,
                input: input.to_string(),
                seed: 419,
                width: (b[2] - b[0]) * scale + 0.5,
                height: 1.5,
                pen_width: Some(ink_layout::PEN_WIDTH),
                glyphs: vec![ink_layout::Glyph {
                    run_labels: vec![],
                    condition_label: Some(input.to_string()),
                    label: input.to_string(),
                    mode: "neural-sequence-probe".into(),
                    model_id: d.model_id.clone(),
                    decoder_sha256: checksum.into(),
                    latent_seed: 419,
                    strokes,
                    transform: [scale, scale, 0.25 - b[0] * scale, 0.25 - b[1] * scale],
                }],
                procedures: vec![],
            };
            let path = folder.join(format!("{i}.svg"));
            fs::write(&path, ink_layout::svg(&drawing)?)?;
            Ok(
                json!({"input":input,"svg":path,"terminated":true,"component":"whole-sample sequence decoder; visual quality unscored"}),
            )
        })();
        match generated {
            Ok(v) => probes.push(v),
            Err(e) => probes.push(json!({"input":input,"error":e.to_string()})),
        }
    }
    Ok(probes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pen_lift_position_loss_is_not_diluted_by_dense_curves() {
        let error = |curves: usize| {
            let sample = Sample {
                sample_id: "move-density-fixture".into(),
                label: "x".into(),
                source_sha256: "a".repeat(64),
                split: "train".into(),
                metadata: Value::Null,
                strokes: vec![
                    ink_corpus::Stroke {
                        points: (0..=3 * curves)
                            .map(|t| {
                                let a = t as f32 / (3 * curves) as f32 * std::f32::consts::TAU;
                                [a.cos(), a.sin()]
                            })
                            .collect(),
                        width: 0.02,
                        source_index: 0,
                        uuid: "fixture".into(),
                    },
                    ink_corpus::Stroke {
                        points: vec![[3., 0.]],
                        width: 0.02,
                        source_index: 1,
                        uuid: "short-native-tap".into(),
                    },
                ],
            };
            let rows = prepare(vec![sample], &["\0".into(), "x".into()]).unwrap();
            let b = batch(&[&rows[0]], Device::Cpu, false, 0);
            let predicted = b.events.narrow(-1, 0, 7) + 0.;
            let target = b.events.double_value(&[0, 0, 0]);
            let _ = predicted
                .get(0)
                .get(0)
                .get(0)
                .fill_(target + 0.1 / b.position_scale.double_value(&[0, 0, 0]));
            local_geometry(&predicted, &b).double_value(&[])
        };
        let sparse = error(12);
        let dense = error(120);
        assert!(
            (sparse - dense).abs() < 1e-5,
            "same pen-lift position error weakened by dense native curves: {sparse} vs {dense}"
        );
        assert!(
            sparse > 0.005,
            "pen-lift position received no meaningful supervision"
        );
    }
    #[test]
    fn path_error_is_measured_in_source_height_not_sentence_width() {
        let physical_error = Tensor::from_slice(&[0.1_f32, 0.1, 0.2, 0.2]).reshape([1, 2, 2]);
        let zeros = Tensor::zeros_like(&physical_error);
        let mask = Tensor::ones([1, 2], (Kind::Float, Device::Cpu));
        let one = Tensor::ones([1, 1, 1], (Kind::Float, Device::Cpu));
        let short = trajectory_geometry(&physical_error, &zeros, &mask, &one).double_value(&[]);
        let long = trajectory_geometry(&(&physical_error / 10.), &zeros, &mask, &(&one * 10.))
            .double_value(&[]);
        assert!(
            (short - long).abs() < 1e-6,
            "same physical path error weakened for wide text: {short} vs {long}"
        );
    }
    #[test]
    fn legacy_input_transfer_preserves_predictions_and_sequence_snapshot() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let vocabulary = vec!["\0".into(), "x".into(), "y".into()];
        let old = Network::with_conditioning(Device::Cpu, vocabulary.len(), false);
        let decoder = old.export("transfer-fixture", &vocabulary, 64).unwrap();
        let directory =
            std::env::temp_dir().join(format!("ink-global-transfer-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        fs::create_dir(&directory).unwrap();
        save(&directory.join("decoder.json"), &decoder).unwrap();
        save(&directory.join("vocabulary.json"), &vocabulary).unwrap();
        save(&directory.join("train-lineage.json"), &json!([])).unwrap();
        old.store
            .save(directory.join("checkpoint.safetensors"))
            .unwrap();
        let mut manifest = json!({"id":"transfer-fixture","kind":"sequence-cvae-v3","approved_for_default":true,"quality_review_sha256":"old-review","verification_sha256":"old-verification","recipe":serde_json::from_str::<Value>(include_str!("../../../recipes/sequence-cvae-v10.json")).unwrap()});
        for (key, file) in [
            ("decoder", "decoder.json"),
            ("vocabulary", "vocabulary.json"),
            ("checkpoint", "checkpoint.safetensors"),
            ("train_lineage", "train-lineage.json"),
        ] {
            manifest[format!("{key}_sha256")] = json!(file_hash(&directory.join(file)).unwrap());
        }
        save(&directory.join("manifest.json"), &manifest).unwrap();
        let new = Network::new(Device::Cpu, vocabulary.len());
        transfer_parent(&new, &directory, &vocabulary, &[]).unwrap();
        let states = Tensor::randn([2, 7, HIDDEN as i64], (Kind::Float, Device::Cpu));
        let previous = Tensor::randn([2, 7, FEATURES as i64], (Kind::Float, Device::Cpu));
        let style = Tensor::randn([2, 7, STYLE as i64], (Kind::Float, Device::Cpu));
        let progress = Tensor::randn([2, 7, 1], (Kind::Float, Device::Cpu));
        let global = Tensor::randn([2, 7, HIDDEN as i64], (Kind::Float, Device::Cpu));
        let count = Tensor::randn([2, 7, 1], (Kind::Float, Device::Cpu));
        tch::no_grad(|| {
            let (a, _) = old
                .decoder
                .seq(&Tensor::cat(&[&previous, &states, &style, &progress], -1));
            let (b, _) = new.decoder.seq(&Tensor::cat(
                &[&previous, &states, &global, &style, &progress, &count],
                -1,
            ));
            assert!(
                (a - b).abs().max().double_value(&[]) < 1e-6,
                "conditioning expansion reinterpreted old input columns"
            );
        });
        let out = directory.join("frozen");
        super::super::snapshot(&directory, &out).unwrap();
        assert_eq!(
            fs::read(out.join("decoder.json")).unwrap(),
            fs::read(directory.join("decoder.json")).unwrap()
        );
        let frozen = super::super::show(&out).unwrap();
        assert_eq!(frozen["approved_for_default"], false);
        assert!(frozen.get("quality_review_sha256").is_none());
        assert!(frozen.get("verification_sha256").is_none());
        assert!(super::super::snapshot(&directory, &out).is_err());
        fs::write(directory.join("decoder.json"), b"{}").unwrap();
        let rejected = directory.join("invalid");
        assert!(super::super::snapshot(&directory, &rejected).is_err());
        assert!(!rejected.exists());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn canonical_content_warmup_precedes_autonomous_rollout() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let vocabulary = vec!["\0".into(), "x".into()];
        let rows = prepare(
            vec![Sample {
                sample_id: "warmup-fixture".into(),
                label: "x".into(),
                source_sha256: "a".repeat(64),
                split: "train".into(),
                metadata: Value::Null,
                strokes: vec![ink_corpus::Stroke {
                    points: vec![[0., 0.], [0.2, 0.1], [0.4, 0.6], [0.8, 0.2]],
                    width: 0.02,
                    source_index: 0,
                    uuid: "fixture".into(),
                }],
            }],
            &vocabulary,
        )
        .unwrap();
        let mut net = Network::new(Device::Cpu, vocabulary.len());
        tch::no_grad(|| {
            let _ = net.mu.ws.zero_();
            let _ = net.mu.bs.as_mut().unwrap().zero_();
        });
        let b = batch(&[&rows[0]], Device::Cpu, false, 0);
        tch::no_grad(|| {
            for step in [1500, 1501, 1502, 9999] {
                let canonical = net.loss(&b, step, false).double_value(&[]);
                let training = net.loss(&b, step, true).double_value(&[]);
                assert!(
                    (canonical - training).abs() < 1e-6,
                    "content warmup perturbed source feedback/style at step {step}: {canonical} vs {training}"
                );
            }
        });
    }
    #[test]
    fn token_window_native_export_parity_with_padded_text() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let vocabulary = vec!["\0".into(), "x".into(), "y".into()];
        let rows = prepare(
            ["x".repeat(20), "xy".repeat(40)]
                .into_iter()
                .enumerate()
                .map(|(i, label)| Sample {
                    sample_id: format!("window-{i}"),
                    label,
                    source_sha256: format!("{i:064x}"),
                    split: "train".into(),
                    metadata: Value::Null,
                    strokes: vec![ink_corpus::Stroke {
                        points: (0..=3 * if i == 0 { 64 } else { 400 })
                            .map(|t| [t as f32 / 300., (t as f32 / 10.).sin()])
                            .collect(),
                        width: 0.02,
                        source_index: 0,
                        uuid: "fixture".into(),
                    }],
                })
                .collect(),
            &vocabulary,
        )
        .unwrap();
        let net = Network::new(Device::Cpu, vocabulary.len());
        let d = net.export("window-fixture", &vocabulary, 64).unwrap();
        let b = batch(&rows.iter().collect::<Vec<_>>(), Device::Cpu, false, 0);
        let z = Tensor::zeros([2, STYLE as i64], (Kind::Float, Device::Cpu));
        tch::no_grad(|| {
            let y = net.forward(&b, &z);
            let scheduled = net.forward_scheduled(&b, &z, 0);
            let error = (&y - scheduled).abs().max().double_value(&[]);
            assert!(
                error < 1e-5,
                "long teacher/scheduled alignment mismatch {error}"
            );
            for (i, row) in rows.iter().enumerate() {
                let (states, mean) = d.condition(&row.sample.label).unwrap();
                let mut h = vec![0.; HIDDEN];
                let mut previous = [0.; FEATURES];
                for (t, event) in row.events.iter().enumerate() {
                    let (next, output) = d
                        .step(
                            &previous,
                            &h,
                            &states,
                            &mean,
                            &[0.; STYLE],
                            t as f32 / row.events.len() as f32,
                        )
                        .unwrap();
                    let expected = Vec::<f32>::try_from(y.get(i as i64).get(t as i64)).unwrap();
                    assert!(
                        output
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| (a - b).abs() < 2e-5)
                    );
                    h = next;
                    previous = *event;
                }
            }
        });
    }
    #[test]
    #[ignore = "prepare a separate source-preserving diagnostic subset; never changes active corpus"]
    fn prepare_one_source_diagnostic_corpus() {
        let all = load(Path::new(&std::env::var("INK_PERF_CORPUS").unwrap())).unwrap();
        let mut selected = vec![
            all.iter()
                .find(|s| s.sample_id == "s_75cf18379676be431fd1cba3")
                .unwrap()
                .clone(),
        ];
        for split in ["validation", "test"] {
            selected.push(all.iter().find(|s| s.split == split).unwrap().clone());
        }
        audit(&selected).unwrap();
        let data = selected
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(std::env::var("INK_PERF_OUT").unwrap(), data).unwrap();
    }
    #[test]
    #[ignore = "explicit frozen-checkpoint teacher/free diagnostic; no weight updates"]
    fn frozen_teacher_rollout_diagnostic_without_updates() {
        tch::set_num_threads(2);
        let model = std::path::PathBuf::from(std::env::var("INK_LOSS_MODEL").unwrap());
        let out = std::path::PathBuf::from(std::env::var("INK_DIAG_OUT").unwrap());
        fs::create_dir_all(&out).unwrap();
        let vocabulary: Vec<String> =
            serde_json::from_slice(&fs::read(model.join("vocabulary.json")).unwrap()).unwrap();
        let samples = load(Path::new(&std::env::var("INK_PERF_CORPUS").unwrap()))
            .unwrap()
            .into_iter()
            .filter(|s| {
                s.split == "train"
                    && ["s_75cf18379676be431fd1cba3", "patch-bbce86e4e0c4-001"]
                        .contains(&s.sample_id.as_str())
            })
            .collect();
        let decoder = SequenceDecoder::load(&model.join("decoder.json")).unwrap();
        let rows = prepare_using(
            samples,
            &vocabulary,
            if decoder.em_normalization {
                encode_em
            } else {
                sequence::encode
            },
        )
        .unwrap();
        let mut net = Network::with_architecture(
            Device::Cpu,
            vocabulary.len(),
            decoder.global_conditioning,
            decoder.alignment.is_some(),
        );
        net.store
            .load(model.join("checkpoint.safetensors"))
            .unwrap();
        net.attention_strength = decoder.attention_strength as f64;
        net.local_conditioning = decoder.local_conditioning;
        net.token_space_attention = decoder.token_space_attention;
        let mut reports = vec![];
        for (i, row) in rows.iter().enumerate() {
            let parity_error =
                parity(&net, &decoder, std::slice::from_ref(row), Device::Cpu).unwrap();
            reports.push(json!({"sample":row.sample.sample_id,"full_teacher_native_export_parity":parity_error,"checked_events":row.events.len()}));
            let b = batch(&[row], Device::Cpu, false, 0);
            let (mu, _) = tch::no_grad(|| net.posterior(&b));
            for policy in 0..5 {
                let (strokes, event_count) = if policy == 0 {
                    (
                        sequence::decode(&row.events, [0., 0.], 1.).unwrap(),
                        row.events.len(),
                    )
                } else if policy == 4 {
                    let strokes = decoder
                        .generate_tokens(
                            &ink_layout::tokenize(&row.sample.label).unwrap(),
                            419,
                            0.,
                            419,
                        )
                        .unwrap();
                    let count = strokes.iter().map(|s| (s.points.len() - 1) / 3 + 1).sum();
                    (strokes, count)
                } else {
                    let z = if policy == 3 {
                        mu.shallow_clone()
                    } else {
                        Tensor::zeros_like(&mu)
                    };
                    let y = tch::no_grad(|| net.forward(&b, &z));
                    let mut events = vec![];
                    for (t, original) in row.events.iter().enumerate() {
                        let values = Vec::<f32>::try_from(y.get(0).get(t as i64)).unwrap();
                        let kind = if t == 0 {
                            0
                        } else if policy >= 2 {
                            original[7..10].iter().position(|&k| k > 0.5).unwrap()
                        } else {
                            values[..3]
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.total_cmp(b.1))
                                .unwrap()
                                .0
                        };
                        let mut e = [0.; FEATURES];
                        e[7 + kind] = 1.;
                        if kind == 2 {
                            events.push(e);
                            break;
                        }
                        let mix = values[3..3 + MIXTURES]
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0;
                        e[..7].copy_from_slice(
                            &values[3 + MIXTURES + mix * 14..3 + MIXTURES + mix * 14 + 7],
                        );
                        events.push(e);
                    }
                    if events.last().unwrap()[END] < 0.5 {
                        let mut end = [0.; FEATURES];
                        end[END] = 1.;
                        events.push(end);
                    }
                    (
                        sequence::decode(&events, [0., 0.], 1.).unwrap(),
                        events.len(),
                    )
                };
                let count = strokes.len();
                let mut labels = vec![];
                ink_layout::labels(&ink_layout::parse(&row.sample.label).unwrap(), &mut labels);
                let glyph = ink_layout::Glyph {
                    run_labels: labels,
                    condition_label: None,
                    label: row.sample.label.clone(),
                    mode: "diagnostic-source-guided".into(),
                    model_id: decoder.model_id.clone(),
                    decoder_sha256: "diagnostic".into(),
                    latent_seed: 419,
                    strokes,
                    transform: [1., 1., 0., 0.],
                };
                let drawing = ink_layout::generate_sequence(&row.sample.label, 419, glyph).unwrap();
                fs::write(
                    out.join(format!("{i}-{policy}.svg")),
                    ink_layout::svg(&drawing).unwrap(),
                )
                .unwrap();
                reports.push(json!({"sample":row.sample.sample_id,"policy":policy,"strokes":count,"events":event_count}));
            }
        }
        save(&out.join("teacher-report.json"), &reports).unwrap();
        println!("{}", json!(reports));
    }
    #[test]
    #[ignore = "frozen train-only text ablation; no optimizer or data mutation"]
    fn frozen_text_condition_ablation_without_updates() {
        tch::set_num_threads(2);
        let model = std::path::PathBuf::from(std::env::var("INK_LOSS_MODEL").unwrap());
        let decoder = SequenceDecoder::load(&model.join("decoder.json")).unwrap();
        let manifest = super::super::show(&model).unwrap();
        let checksum = file_hash(&model.join("checkpoint.safetensors")).unwrap();
        assert_eq!(manifest["checkpoint_sha256"], checksum);
        let samples = load(Path::new(&std::env::var("INK_PERF_CORPUS").unwrap()))
            .unwrap()
            .into_iter()
            .filter(|s| s.split == "train")
            .collect();
        let mut rows = prepare_using(
            samples,
            &decoder.vocabulary,
            if decoder.em_normalization {
                encode_em
            } else {
                sequence::encode
            },
        )
        .unwrap();
        rows.retain(|r| r.tokens.len() > 2 && r.events.len() > 2);
        let mut net = Network::with_architecture(
            Device::Cpu,
            decoder.vocabulary.len(),
            decoder.global_conditioning,
            decoder.alignment.is_some(),
        );
        net.attention_strength = decoder.attention_strength as f64;
        net.local_conditioning = decoder.local_conditioning;
        net.token_space_attention = decoder.token_space_attention;
        net.store
            .load(model.join("checkpoint.safetensors"))
            .unwrap();
        let mut reports = vec![];
        tch::no_grad(|| {
            for row in &rows {
                let b = batch(&[row], Device::Cpu, false, 0);
                let z = Tensor::zeros([1, STYLE as i64], (Kind::Float, Device::Cpu));
                let score = |y: Tensor| {
                    let kinds = b.events.narrow(-1, 7, 3);
                    let ce = -(kinds * y.narrow(-1, 0, 3).log_softmax(-1, Kind::Float))
                        .sum(Kind::Float)
                        / b.mask.sum(Kind::Float);
                    let component = y.narrow(-1, 3, MIXTURES as i64).argmax(-1, false);
                    let means = y
                        .narrow(-1, (3 + MIXTURES) as i64, (MIXTURES * 14) as i64)
                        .reshape([1, -1, MIXTURES as i64, 14]);
                    let predicted = means
                        .gather(
                            2,
                            &component
                                .unsqueeze(-1)
                                .unsqueeze(-1)
                                .expand([-1, -1, 1, 6], true),
                            false,
                        )
                        .squeeze_dim(2);
                    let dims = Tensor::cat(
                        &[
                            b.events.narrow(-1, 7, 1) + b.events.narrow(-1, 8, 1),
                            b.events.narrow(-1, 7, 1) + b.events.narrow(-1, 8, 1),
                            b.events.narrow(-1, 8, 1).repeat([1, 1, 4]),
                        ],
                        -1,
                    );
                    let squared =
                        ((predicted - b.events.narrow(-1, 0, 6)) * &b.position_scale).square();
                    let by_class = |kind: i64, dimensions: i64| {
                        let mask = b.events.narrow(-1, kind, 1);
                        (squared.narrow(-1, 0, dimensions) * &mask).sum(Kind::Float)
                            / (mask.sum(Kind::Float) * dimensions).clamp_min(1.)
                    };
                    let geometry =
                        (&squared * &dims).sum(Kind::Float) / dims.sum(Kind::Float).clamp_min(1.);
                    (
                        ce.double_value(&[]),
                        geometry.double_value(&[]),
                        by_class(7, 2).double_value(&[]),
                        by_class(8, 6).double_value(&[]),
                    )
                };
                let correct = score(net.forward(&b, &z));
                let mut shuffled = batch(&[row], Device::Cpu, false, 0);
                shuffled.tokens = shuffled.tokens.flip([1]);
                let reversed = score(net.forward(&shuffled, &z));
                reports.push(json!({"sample":row.sample.sample_id,"label":row.sample.label,"correct_event_ce":correct.0,"reversed_event_ce":reversed.0,"correct_em_mse":correct.1,"reversed_em_mse":reversed.1,"correct_move_mse":correct.2,"correct_curve_mse":correct.3}));
            }
        });
        let count = reports.len() as f64;
        let mean = |key: &str| {
            reports
                .iter()
                .map(|r| r[key].as_f64().unwrap())
                .sum::<f64>()
                / count
        };
        let report = json!({"decoder_sha256":file_hash(&model.join("decoder.json")).unwrap(),"checkpoint_sha256":checksum,"optimizer_updates":0,"canonical_zero_style":true,"source_guided":true,"scope":"train-only text ablation, never relabels a source or certifies free recognition","samples":reports.len(),"correct_event_ce":mean("correct_event_ce"),"reversed_event_ce":mean("reversed_event_ce"),"correct_em_mse":mean("correct_em_mse"),"reversed_em_mse":mean("reversed_em_mse"),"correct_move_mse":mean("correct_move_mse"),"correct_curve_mse":mean("correct_curve_mse"),"rows":reports});
        save(Path::new(&std::env::var("INK_PERF_OUT").unwrap()), &report).unwrap();
        assert_eq!(
            checksum,
            file_hash(&model.join("checkpoint.safetensors")).unwrap()
        );
        println!("{}", report);
    }
    #[test]
    #[ignore = "explicit checkpoint diagnostic; forward only, no weight updates"]
    fn checkpoint_loss_diagnostic_without_updates() {
        tch::set_num_threads(2);
        let (device, _) = super::super::rocm::device(None).unwrap();
        let model = std::env::var("INK_LOSS_MODEL").unwrap();
        let model = Path::new(&model);
        let manifest = super::super::show(model).unwrap();
        assert_eq!(
            manifest["recipe"],
            serde_json::from_str::<Value>(RECIPE).unwrap(),
            "this diagnostic measures the current training objective; historical recipes need their archived diagnostic"
        );
        assert_eq!(
            manifest["checkpoint_sha256"],
            file_hash(&model.join("checkpoint.safetensors")).unwrap()
        );
        let vocab: Vec<String> =
            serde_json::from_slice(&fs::read(model.join("vocabulary.json")).unwrap()).unwrap();
        let samples: Vec<_> = load(Path::new(&std::env::var("INK_PERF_CORPUS").unwrap()))
            .unwrap()
            .into_iter()
            .filter(|s| s.split == "train")
            .collect();
        let rows = prepare(samples, &vocab).unwrap();
        let execution = super::super::execution::ExecutionPlan::resume(&manifest).unwrap();
        let groups = execution.groups(&rows.iter().map(|r| r.events.len()).collect::<Vec<_>>());
        let mut net = Network::new(device, vocab.len());
        net.store
            .load(model.join("checkpoint.safetensors"))
            .unwrap();
        let mut results = vec![];
        let mut sum = 0.;
        let mut teacher_sum = 0.;
        let epoch_start =
            (manifest["updates"].as_u64().unwrap() / groups.len() as u64 + 1) * groups.len() as u64;
        tch::no_grad(|| {
            for (slot, indices) in groups.iter().enumerate() {
                let selected: Vec<_> = indices.iter().map(|&i| &rows[i]).collect();
                let b = batch(&selected, device, false, 0);
                let step = epoch_start + slot as u64;
                tch::manual_seed((419 + step) as i64);
                let terms = net.loss_components(&b, step, true);
                let loss = terms.total().double_value(&[]);
                let teacher = net.loss(&b, step, false).double_value(&[]);
                sum += loss * selected.len() as f64;
                teacher_sum += teacher * selected.len() as f64;
                let max_width = selected
                    .iter()
                    .flat_map(|r| &r.events)
                    .map(|e| e[6])
                    .fold(0_f32, f32::max);
                results.push(json!({"slot":slot,"loss":loss,"teacher_loss":teacher,"scheduled":step.is_multiple_of(2),
                    "reconstruction":terms.reconstruction.double_value(&[]),"kl":terms.kl.double_value(&[]),
                    "duration":terms.duration.double_value(&[]),"absolute":terms.absolute.double_value(&[]),
                    "geometry":terms.geometry.double_value(&[]),
                    "max_normalized_width":max_width,"rows": selected.iter().map(|r| json!({"id":r.sample.sample_id,"label":r.sample.label,"events":r.events.len(),"max_width":r.events.iter().map(|e| e[6]).fold(0_f32,f32::max)})).collect::<Vec<_>>() }));
            }
        });
        results.sort_by(|a, b| {
            b["loss"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["loss"].as_f64().unwrap())
        });
        let report = json!({"model":model,"checkpoint_sha256":manifest["checkpoint_sha256"],"trained_recipe":manifest["recipe"]["id"],"input_recipe":serde_json::from_str::<Value>(RECIPE).unwrap()["id"],"optimizer_updates":0,
            "samples":rows.len(),"weighted_mean":sum / rows.len() as f64,"teacher_mean":teacher_sum / rows.len() as f64,"batches_descending":results});
        save(Path::new(&std::env::var("INK_PERF_OUT").unwrap()), &report).unwrap();
        println!(
            "mean: {}; teacher: {}; worst: {}",
            report["weighted_mean"], report["teacher_mean"], report["batches_descending"][0]
        );
        assert_eq!(
            manifest["checkpoint_sha256"],
            file_hash(&model.join("checkpoint.safetensors")).unwrap()
        );
        assert!(
            report["weighted_mean"].as_f64().unwrap().abs() < 1000.,
            "checkpoint epoch loss exploded: {}",
            report["weighted_mean"]
        );
    }
    #[test]
    #[ignore = "explicit GPU benchmark; gradients only, never optimizer updates"]
    fn gpu_throughput_without_weight_updates() {
        tch::set_num_threads(2);
        let (device, hardware) = super::super::rocm::device(None).unwrap();
        let Device::Cuda(gpu_index) = device else {
            panic!("GPU benchmark requires HIP");
        };
        let corpus = std::env::var("INK_PERF_CORPUS").unwrap();
        let samples: Vec<_> = load(Path::new(&corpus))
            .unwrap()
            .into_iter()
            .filter(|s| s.split == "train")
            .collect();
        let mut vocab = BTreeSet::from(["\0".to_owned()]);
        for s in &samples {
            vocab.extend(ink_layout::tokenize(&s.label).unwrap());
        }
        let vocab: Vec<_> = vocab.into_iter().collect();
        let rows = prepare(samples, &vocab).unwrap();
        let mut order: Vec<_> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].events.len());
        tch::manual_seed(419);
        let net = Network::new(device, vocab.len());
        let fingerprint = || -> BTreeMap<String, Vec<f32>> {
            net.store
                .variables()
                .into_iter()
                .map(|(name, weight)| {
                    (
                        name,
                        Vec::<f32>::try_from(weight.to_device(Device::Cpu).reshape([-1])).unwrap(),
                    )
                })
                .collect()
        };
        let before = fingerprint();
        let mut timings = vec![];
        for n in [8, 32, 64] {
            let offset = order.len() / 2;
            let chosen: Vec<_> = order[offset..offset + n]
                .iter()
                .map(|&i| &rows[i])
                .collect();
            let b = batch(&chosen, device, false, 0);
            for scheduled in [false, true] {
                let step = if scheduled { 2003 } else { 2000 };
                // Warm-up and timed repetitions compute gradients but cannot
                // modify any model tensor: no optimizer is constructed here.
                net.loss(&b, step, true).backward();
                for mut w in net.store.trainable_variables() {
                    w.zero_grad();
                }
                tch::Cuda::synchronize(gpu_index as i64);
                let started = Instant::now();
                for _ in 0..3 {
                    net.loss(&b, step, true).backward();
                    for mut w in net.store.trainable_variables() {
                        w.zero_grad();
                    }
                }
                tch::Cuda::synchronize(gpu_index as i64);
                let seconds = started.elapsed().as_secs_f64() / 3.;
                timings.push(json!({"batch":n,"events":b.events.size()[1],"scheduled":scheduled,"forward_backward_seconds":seconds,"samples_per_second":n as f64/seconds}));
            }
        }
        let mut epochs = vec![];
        if std::env::var_os("INK_PERF_EPOCH").is_some() {
            for (name, execution, cache) in [
                (
                    "legacy-eight",
                    super::super::execution::ExecutionPlan::resume(&json!({})).unwrap(),
                    false,
                ),
                (
                    "gpu-throughput",
                    super::super::execution::ExecutionPlan::new(false),
                    true,
                ),
            ] {
                let groups =
                    execution.groups(&rows.iter().map(|r| r.events.len()).collect::<Vec<_>>());
                let cached = cache.then(|| {
                    groups
                        .iter()
                        .map(|g| {
                            batch(
                                &g.iter().map(|&i| &rows[i]).collect::<Vec<_>>(),
                                device,
                                false,
                                0,
                            )
                        })
                        .collect::<Vec<_>>()
                });
                let mut order: Vec<_> = (0..groups.len()).collect();
                let mut rng = 419_u64;
                for i in (1..order.len()).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    order.swap(i, rng as usize % (i + 1));
                }
                tch::Cuda::synchronize(gpu_index as i64);
                let started = Instant::now();
                for (step, &i) in order.iter().enumerate() {
                    let prepared;
                    let b = if let Some(cached) = &cached {
                        &cached[i]
                    } else {
                        prepared = batch(
                            &groups[i].iter().map(|&i| &rows[i]).collect::<Vec<_>>(),
                            device,
                            false,
                            0,
                        );
                        &prepared
                    };
                    let loss = net.loss(b, 2000 + step as u64, true);
                    assert!(loss.double_value(&[]).is_finite());
                    loss.backward();
                    for mut w in net.store.trainable_variables() {
                        w.zero_grad();
                    }
                }
                tch::Cuda::synchronize(gpu_index as i64);
                let seconds = started.elapsed().as_secs_f64();
                epochs.push(json!({"name":name,"batches":groups.len(),"samples":rows.len(),"seconds":seconds,"samples_per_second":rows.len() as f64/seconds,"cached_inputs":cache,"execution_plan":execution}));
                println!("Epoch benchmark {name}: {seconds:.3}s");
            }
        }
        let after = fingerprint();
        assert_eq!(before, after);
        let report = json!({"hardware":hardware,"optimizer_updates":0,"all_weights_unchanged":true,"timings":timings,"epochs":epochs});
        save(Path::new(&std::env::var("INK_PERF_OUT").unwrap()), &report).unwrap();
        println!("{report}");
    }
    #[test]
    fn scheduled_forward_matches_teacher_at_zero_probability_without_updates() {
        tch::manual_seed(419);
        let device = Device::Cpu;
        let vocabulary = vec!["\0".into(), "T".into(), "t".into()];
        let sample = Sample {
            sample_id: "forward-only".into(),
            label: "Tt".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: Value::Null,
            strokes: vec![ink_corpus::Stroke {
                points: vec![[0., 0.], [0.2, 0.3], [0.5, 0.9], [0.8, 0.6]],
                width: 0.02,
                source_index: 0,
                uuid: "native".into(),
            }],
        };
        let rows = prepare(vec![sample], &vocabulary).unwrap();
        let b = batch(&[&rows[0]], device, false, 0);
        let net = Network::new(device, vocabulary.len());
        let z = Tensor::zeros([1, STYLE as i64], (Kind::Float, device));
        tch::no_grad(|| {
            let direct = net.forward(&b, &z);
            let recurrent = net.forward_scheduled(&b, &z, 0);
            assert!((direct - recurrent).abs().max().double_value(&[]) < 1e-5);
            assert!(
                net.forward_scheduled(&b, &z, 2003)
                    .isfinite()
                    .all()
                    .int64_value(&[])
                    != 0
            );
            assert!(net.loss(&b, 2003, true).double_value(&[]).is_finite());
            // Reusing immutable prepared inputs must preserve source geometry,
            // masks, event weights and previous-event shifts at later updates.
            let rebuilt = batch(&[&rows[0]], device, false, 2003);
            for (cached, fresh) in [
                (&b.tokens, &rebuilt.tokens),
                (&b.text_mask, &rebuilt.text_mask),
                (&b.events, &rebuilt.events),
                (&b.mask, &rebuilt.mask),
                (&b.weights, &rebuilt.weights),
                (&b.previous, &rebuilt.previous),
            ] {
                assert!(cached.equal(fresh));
            }
            assert_eq!(
                net.loss(&b, 2000, false).double_value(&[]),
                net.loss(&rebuilt, 2000, false).double_value(&[])
            );
        });
    }
    #[test]
    fn actual_sequence_update_optimizer_resume_and_export() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let device = Device::Cpu;
        let vocabulary = vec!["\0".into(), "x".into(), "y".into()];
        let sample = Sample {
            sample_id: "fixture".into(),
            label: "xy".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: Value::Null,
            strokes: vec![ink_corpus::Stroke {
                points: vec![[0., 0.], [0.1, 0.2], [0.2, 0.3], [0.4, 0.5]],
                source_index: 0,
                uuid: "s".into(),
                width: 0.01,
            }],
        };
        let rows = prepare(vec![sample], &vocabulary).unwrap();
        let net = Network::new(device, vocabulary.len());
        let mut adam = Adam::new();
        let b = batch(&[&rows[0]], device, false, 0);
        let before = net.export("fixture", &vocabulary, 128).unwrap();
        let loss = net.loss(&b, adam.step, false);
        loss.backward();
        adam.update(&net.store).unwrap();
        let after = net.export("fixture", &vocabulary, 128).unwrap();
        assert_ne!(before.output.weight, after.output.weight);
        assert!(parity(&net, &after, &rows, device).unwrap() < 1e-5);
        let dir = std::env::temp_dir().join(format!("ink-sequence-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        net.store.save(dir.join("checkpoint.safetensors")).unwrap();
        adam.save(&dir.join("optimizer.safetensors")).unwrap();
        let mut restored = Network::new(device, vocabulary.len());
        restored
            .store
            .load(dir.join("checkpoint.safetensors"))
            .unwrap();
        let mut resumed = Adam::restore(
            &dir.join("optimizer.safetensors"),
            adam.step,
            &restored.store,
        )
        .unwrap();
        let a = net.loss(&b, adam.step, false);
        let c = restored.loss(&b, resumed.step, false);
        assert_eq!(a.double_value(&[]), c.double_value(&[]));
        a.backward();
        c.backward();
        adam.update(&net.store).unwrap();
        resumed.update(&restored.store).unwrap();
        let a = net.export("fixture", &vocabulary, 128).unwrap();
        let c = restored.export("fixture", &vocabulary, 128).unwrap();
        assert_eq!(a.output.weight, c.output.weight);
        fs::remove_dir_all(dir).unwrap();
    }
}

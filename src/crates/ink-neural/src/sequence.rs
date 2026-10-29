//! Inference-only attention GRU cubic decoder; exported weights, never source ink.
use crate::{Linear, normal};
use anyhow::{Context, Result, ensure};
use ink_corpus::{
    Stroke,
    sequence::{CURVE, END, FEATURES, MOVE},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};
pub const HIDDEN: usize = 128;
pub const EMBED: usize = 48;
pub const STYLE: usize = 16;
pub const MIXTURES: usize = 5;
pub const PARAMETERS: usize = 3 + MIXTURES * 15;
pub fn decoder_input_width(global_conditioning: bool) -> usize {
    FEATURES + HIDDEN + STYLE + 1 + if global_conditioning { HIDDEN + 1 } else { 0 }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Gru {
    pub input: Linear,
    pub recurrent: Linear,
}

#[cfg(test)]
mod whole_run_contracts {
    use super::*;
    #[test]
    fn first_event_receives_complete_prompt_context() {
        let mut d = fixture();
        d.decoder.input.bias[2 * HIDDEN] = 0.;
        d.decoder.input.weight[2 * HIDDEN * d.decoder.input.input + FEATURES + HIDDEN] = 2.;
        let zero = vec![0.; HIDDEN];
        let states = vec![vec![1.; HIDDEN]; 20];
        let (a, _) = d
            .step(
                &[0.; FEATURES],
                &zero,
                &states,
                &vec![1.; HIDDEN],
                &[0.; STYLE],
                0.,
            )
            .unwrap();
        let (b, _) = d
            .step(
                &[0.; FEATURES],
                &zero,
                &states,
                &vec![-1.; HIDDEN],
                &[0.; STYLE],
                0.,
            )
            .unwrap();
        assert!(
            a[0] > b[0] + 0.2,
            "first event cannot distinguish complete prompts with the same local prefix"
        );
    }
    #[test]
    fn local_window_keeps_character_resolution_in_long_text() {
        let mut d = fixture();
        d.decoder.input.bias[2 * HIDDEN] = 0.;
        d.decoder.input.weight[2 * HIDDEN * d.decoder.input.input + FEATURES] = 2.;
        let zero = vec![0.; HIDDEN];
        for count in [20, 80] {
            let mut states = vec![vec![-1.; HIDDEN]; count];
            states[count / 2].fill(1.);
            let (hidden, _) = d
                .step(&[0.; FEATURES], &zero, &states, &zero, &[0.; STYLE], 0.5)
                .unwrap();
            assert!(
                hidden[0] > 0.2,
                "local character blurred across {count} tokens: {}",
                hidden[0]
            );
        }
    }
    #[test]
    fn learned_window_can_hold_a_letter_without_the_external_clock() {
        let mut d = fixture();
        let mut window = linear(HIDDEN, 3 * WINDOWS);
        window.bias[WINDOWS..2 * WINDOWS].fill(16_f32.ln());
        window.bias[2 * WINDOWS..].fill(0.1_f32.ln());
        d.alignment = Some(Alignment {
            recurrent: Gru {
                input: linear(ALIGN_INPUT, 3 * HIDDEN),
                recurrent: linear(HIDDEN, 3 * HIDDEN),
            },
            window,
        });
        d.decoder.input.bias[2 * HIDDEN] = 0.;
        d.decoder.input.weight[2 * HIDDEN * d.decoder.input.input + FEATURES] = 2.;
        let mut states = vec![vec![-1.; HIDDEN]; 20];
        states[0].fill(1.);
        let zero = vec![0.; HIDDEN];
        let mut h = zero.clone();
        for _ in 0..4 {
            let (a, _) = d
                .step(&[0.; FEATURES], &h, &states, &zero, &[0.; STYLE], 0.)
                .unwrap();
            let (b, _) = d
                .step(&[0.; FEATURES], &h, &states, &zero, &[0.; STYLE], 0.99)
                .unwrap();
            assert_eq!(
                a, b,
                "learned alignment must not be overwritten by duration estimates"
            );
            assert!(a[0] > 0.2, "slow pen lost its current letter");
            h = a;
        }
        let serialized = serde_json::to_vec(&d).unwrap();
        let restored: SequenceDecoder = serde_json::from_slice(&serialized).unwrap();
        restored.validate().unwrap();
        assert_eq!(
            d.step(&[0.; FEATURES], &h, &states, &zero, &[0.; STYLE], 0.4)
                .unwrap(),
            restored
                .step(&[0.; FEATURES], &h, &states, &zero, &[0.; STYLE], 0.4)
                .unwrap()
        );
    }
    #[test]
    fn recurrent_hidden_state_receives_the_current_text_window() {
        let mut d = fixture();
        d.decoder.input.weight[2 * HIDDEN * d.decoder.input.input + FEATURES] = 2.;
        let states = vec![vec![1.; HIDDEN], vec![-1.; HIDDEN]];
        let zero = vec![0.; HIDDEN];
        let (left, _) = d
            .step(&[0.; FEATURES], &zero, &states, &zero, &[0.; STYLE], 0.)
            .unwrap();
        let (right, _) = d
            .step(&[0.; FEATURES], &zero, &states, &zero, &[0.; STYLE], 1.)
            .unwrap();
        assert!(
            left[0] > right[0] + 0.1,
            "current character never reached recurrent state"
        );
        d.local_conditioning = false;
        let (legacy_left, _) = d
            .step(&[0.; FEATURES], &zero, &states, &zero, &[0.; STYLE], 0.)
            .unwrap();
        let (legacy_right, _) = d
            .step(&[0.; FEATURES], &zero, &states, &zero, &[0.; STYLE], 1.)
            .unwrap();
        assert_eq!(
            legacy_left, legacy_right,
            "historical sentence-mean semantics must remain available"
        );
    }
    #[test]
    #[ignore = "explicit frozen-decoder rollout diagnosis; inference only"]
    fn frozen_rollout_diagnostic_without_updates() {
        let decoder =
            SequenceDecoder::load(Path::new(&std::env::var("INK_DIAG_DECODER").unwrap())).unwrap();
        let out = std::path::PathBuf::from(std::env::var("INK_DIAG_OUT").unwrap());
        fs::create_dir_all(&out).unwrap();
        let mut report = vec![];
        for (text_index, text) in [
            "x",
            "Hello, World!",
            "World, Hello!",
            r"\text{There are more of them and they are stronger}",
        ]
        .iter()
        .enumerate()
        {
            let (states, mean) = decoder.condition(text).unwrap();
            let log_length = decoder.length.apply(&mean).unwrap()[0]
                + if decoder.length_per_token {
                    (states.len() as f32).ln()
                } else {
                    0.
                };
            let expected = log_length
                .clamp(0., (decoder.max_events as f32).ln())
                .exp()
                .max(2.);
            for policy in 0..6 {
                let mut hidden = vec![0.; HIDDEN];
                let mut previous = [0.; FEATURES];
                let mut events = vec![];
                let mut move_probability = 0.;
                let mut width_clamps = 0;
                let mut rng = 419_u64;
                let mut uniform = || {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 40) as f32 + 0.5) / 16_777_216.
                };
                for i in 0..decoder.max_events {
                    let (next, output) = decoder
                        .step(
                            &previous,
                            &hidden,
                            &states,
                            &mean,
                            &[0.; STYLE],
                            i as f32 / expected,
                        )
                        .unwrap();
                    hidden = next;
                    let p = softmax(&output[..3]);
                    move_probability += p[0];
                    let categorical = |probabilities: &[f32], u: f32| {
                        let mut sum = 0.;
                        probabilities
                            .iter()
                            .position(|p| {
                                sum += p;
                                sum >= u
                            })
                            .unwrap_or(probabilities.len() - 1)
                    };
                    let kind = if i == 0 {
                        0
                    } else if [2, 3, 5].contains(&policy) {
                        categorical(&p, uniform())
                    } else {
                        p.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0
                    };
                    let mut event = [0.; FEATURES];
                    event[7 + kind] = 1.;
                    if kind == 2 {
                        events.push(event);
                        break;
                    }
                    let probabilities = softmax(&output[3..3 + MIXTURES]);
                    let mixture = if policy >= 4 {
                        (0..MIXTURES)
                            .max_by(|&a, &b| {
                                let score = |m: usize| {
                                    let dims: Vec<usize> = if kind == 0 {
                                        vec![0, 1, 6]
                                    } else {
                                        (0..7).collect()
                                    };
                                    probabilities[m].max(1e-30).ln()
                                        - dims
                                            .iter()
                                            .map(|&j| {
                                                output[3 + MIXTURES + m * 14 + 7 + j].clamp(-9., 0.)
                                            })
                                            .sum::<f32>()
                                };
                                score(a).total_cmp(&score(b))
                            })
                            .unwrap()
                    } else if policy == 3 {
                        categorical(&probabilities, uniform())
                    } else {
                        probabilities
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0
                    };
                    event[..7].copy_from_slice(
                        &output[3 + MIXTURES + mixture * 14..3 + MIXTURES + mixture * 14 + 7],
                    );
                    if event[6] != event[6].abs().clamp(0.0005, 0.08) {
                        width_clamps += 1;
                    }
                    if decoder.canonical_feedback && policy != 1 {
                        event = ink_corpus::sequence::canonical_feedback(event);
                    } else if policy == 0 {
                        event[6] = event[6].abs().clamp(0.0005, 0.08);
                    }
                    previous = event;
                    events.push(event);
                }
                let terminated = events.last().is_some_and(|e| e[END] > 0.5);
                let n = events.len();
                if !terminated {
                    let mut end = [0.; FEATURES];
                    end[END] = 1.;
                    events.push(end);
                }
                let strokes = ink_corpus::sequence::decode(&events, [0., 0.], 1.).unwrap();
                let count = strokes.len();
                let mut labels = vec![];
                ink_layout::labels(&ink_layout::parse(text).unwrap(), &mut labels);
                let glyph = ink_layout::Glyph {
                    run_labels: labels,
                    condition_label: Some(text.to_string()),
                    label: text.to_string(),
                    mode: "diagnostic-frozen-rollout".into(),
                    model_id: decoder.model_id.clone(),
                    decoder_sha256: "diagnostic".into(),
                    latent_seed: 419,
                    strokes,
                    transform: [1., 1., 0., 0.],
                };
                let drawing = ink_layout::generate_sequence(text, 419, glyph).unwrap();
                let path = out.join(format!("{text_index}-{policy}.svg"));
                fs::write(&path, ink_layout::svg(&drawing).unwrap()).unwrap();
                report.push(serde_json::json!({"text":text,"policy":policy,"events":n,"strokes":count,"terminated":terminated,"expected_events":expected,"mean_move_probability":move_probability/n as f32,"width_clamps":width_clamps,"svg":path}));
            }
        }
        fs::write(
            out.join("rollouts.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{}", serde_json::to_string(&report).unwrap());
    }
    fn linear(input: usize, output: usize) -> Linear {
        Linear {
            input,
            weight: vec![0.; input * output],
            bias: vec![0.; output],
        }
    }
    fn fixture() -> SequenceDecoder {
        let mut vocabulary =
            ink_layout::tokenize(r"\boxed{\left[\frac{T}{\sqrt{t}}\right]}").unwrap();
        vocabulary.sort();
        vocabulary.dedup();
        let mut decoder = Gru {
            input: linear(decoder_input_width(true), HIDDEN * 3),
            recurrent: linear(HIDDEN, HIDDEN * 3),
        };
        decoder.input.bias[HIDDEN * 2] = 1.;
        let mut output = linear(HIDDEN * 2 + STYLE, PARAMETERS);
        output.bias[1] = 0.1;
        output.bias[2] = -6.;
        output.weight[2 * output.input] = 10.;
        output.bias[3] = 1.;
        for mode in 0..MIXTURES {
            let offset = 3 + MIXTURES + mode * 14;
            output.bias[offset..offset + 7]
                .copy_from_slice(&[0.1, 0.05, 0.2, 0.12, 0.3, 0.2, 0.025]);
        }
        SequenceDecoder {
            alignment: None,
            em_normalization: false,
            style_domain: "personal".into(),
            whole_run: true,
            length_per_token: true,
            canonical_feedback: true,
            attention_strength: 4.,
            token_space_attention: true,
            global_conditioning: true,
            local_conditioning: true,
            glyph_head: None,
            schema_version: 3,
            kind: "sequence-cvae-v3".into(),
            model_id: "fixture-no-training".into(),
            embedding: vec![vec![0.; EMBED]; vocabulary.len()],
            vocabulary,
            encoder: Gru {
                input: linear(EMBED, HIDDEN * 3),
                recurrent: linear(HIDDEN, HIDDEN * 3),
            },
            decoder,
            query: linear(HIDDEN, HIDDEN),
            output,
            length: linear(HIDDEN, 1),
            max_events: 16,
            metrics: BTreeMap::new(),
        }
    }
    #[test]
    fn whole_run_runtime_has_no_coordinate_noise_or_procedural_math() {
        let d = fixture();
        d.validate().unwrap();
        let a = d
            .generate_tokens(&["T".into(), "t".into()], 419, 1., 419)
            .unwrap();
        let b = d
            .generate_tokens(&["T".into(), "t".into()], 420, 2., 420)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap(),
            "style-independent fixture must not acquire per-coordinate noise"
        );
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].points.len(), 4);
        let runtime = crate::Runtime::Sequence(Box::new(d.clone()));
        let drawing = runtime
            .draw(
                r"\boxed{\left[\frac{T}{\sqrt{t}}\right]}",
                419,
                "fixture",
                1.,
            )
            .unwrap();
        assert_eq!(drawing.glyphs.len(), 1);
        assert!(drawing.procedures.is_empty());
        assert_eq!(ink_layout::drawing_labels(&drawing), ["[", "T", "t", "]"]);
        assert!(runtime.draw("Q", 419, "fixture", 1.).is_err());
        let mut public = d;
        public.style_domain = "public".into();
        let drawing = crate::Runtime::Sequence(Box::new(public))
            .draw("Tt", 419, "fixture", 0.)
            .unwrap();
        assert_eq!(drawing.glyphs[0].mode, "neural-run-public");
    }
    #[test]
    fn historical_decoders_cannot_silently_enable_whole_run_generation() {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value.as_object_mut().unwrap().remove("whole_run");
        value.as_object_mut().unwrap().remove("style_domain");
        value.as_object_mut().unwrap().remove("length_per_token");
        value.as_object_mut().unwrap().remove("canonical_feedback");
        value.as_object_mut().unwrap().remove("attention_strength");
        value.as_object_mut().unwrap().remove("local_conditioning");
        value
            .as_object_mut()
            .unwrap()
            .remove("token_space_attention");
        value.as_object_mut().unwrap().remove("global_conditioning");
        let historical: SequenceDecoder = serde_json::from_value(value).unwrap();
        assert!(!historical.whole_run);
        assert_eq!(historical.style_domain, "personal");
        assert!(!historical.length_per_token);
        assert!(!historical.canonical_feedback);
        assert_eq!(historical.attention_strength, 4.);
        assert!(!historical.local_conditioning);
        assert!(!historical.token_space_attention);
        assert!(!historical.global_conditioning);
    }
}
impl Gru {
    pub fn step(&self, x: &[f32], h: &[f32]) -> Result<Vec<f32>> {
        let a = self.input.apply(x)?;
        let b = self.recurrent.apply(h)?;
        let n = h.len();
        ensure!(a.len() == 3 * n && b.len() == 3 * n, "GRU dimensions");
        Ok((0..n)
            .map(|i| {
                let r = sigmoid(a[i] + b[i]);
                let z = sigmoid(a[n + i] + b[n + i]);
                let candidate = (a[2 * n + i] + r * b[2 * n + i]).tanh();
                (1. - z) * candidate + z * h[i]
            })
            .collect())
    }
}
fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let v: Vec<_> = x.iter().map(|x| (x - m).exp()).collect();
    let sum = v.iter().sum::<f32>();
    v.iter().map(|x| x / sum).collect()
}
#[derive(Clone, Serialize, Deserialize)]
pub struct SequenceDecoder {
    /// Optional learned monotonic window. Absent exports retain their fixed clock.
    #[serde(default)]
    pub alignment: Option<Alignment>,
    #[serde(default)]
    pub em_normalization: bool,
    #[serde(default = "personal_domain")]
    pub style_domain: String,
    /// Set only by the whole-run training recipe. Historical decoders stay glyph based.
    #[serde(default)]
    pub whole_run: bool,
    #[serde(default)]
    pub length_per_token: bool,
    #[serde(default)]
    pub canonical_feedback: bool,
    #[serde(default = "legacy_attention_strength")]
    pub attention_strength: f32,
    /// Keep the window width in characters, independent of complete text length.
    /// Absent on historical exports, whose normalized-progress math is retained.
    #[serde(default)]
    pub token_space_attention: bool,
    /// Whole-run scale requires complete-prompt context at the first event.
    #[serde(default)]
    pub global_conditioning: bool,
    #[serde(default)]
    pub local_conditioning: bool,
    #[serde(default)]
    pub glyph_head: Option<crate::compositional::GlyphDecoder>,
    pub schema_version: u32,
    pub kind: String,
    pub model_id: String,
    pub vocabulary: Vec<String>,
    pub embedding: Vec<Vec<f32>>,
    pub encoder: Gru,
    pub decoder: Gru,
    pub query: Linear,
    pub output: Linear,
    pub length: Linear,
    pub max_events: usize,
    pub metrics: BTreeMap<String, [f32; 3]>,
}
pub const WINDOWS: usize = 3;
pub const ALIGN_INPUT: usize = FEATURES + HIDDEN + STYLE + 1;
#[derive(Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub recurrent: Gru,
    pub window: Linear,
}
fn personal_domain() -> String {
    "personal".into()
}
fn legacy_attention_strength() -> f32 {
    4.
}
impl SequenceDecoder {
    pub fn load(p: &Path) -> Result<Self> {
        let d: Self = serde_json::from_slice(&fs::read(p)?)?;
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.attention_strength.is_finite()
                && self.attention_strength > 0.
                && self.attention_strength <= 512.,
            "invalid attention strength"
        );
        ensure!(
            matches!(self.style_domain.as_str(), "personal" | "public"),
            "invalid style domain"
        );
        if let Some(g) = &self.glyph_head {
            g.validate()?;
        }
        ensure!(
            self.schema_version == 3 && self.kind == "sequence-cvae-v3",
            "unsupported sequence model"
        );
        ensure!(
            self.vocabulary.len() == self.embedding.len()
                && !self.vocabulary.is_empty()
                && self.vocabulary.windows(2).all(|w| w[0] < w[1]),
            "sequence vocabulary invalid"
        );
        ensure!(
            self.max_events > 0 && self.max_events <= 16384,
            "sequence capacity invalid"
        );
        for row in &self.embedding {
            ensure!(
                row.len() == EMBED && row.iter().all(|x| x.is_finite()),
                "embedding invalid"
            );
        }
        let mut layers = vec![
            &self.encoder.input,
            &self.encoder.recurrent,
            &self.decoder.input,
            &self.decoder.recurrent,
            &self.query,
            &self.output,
            &self.length,
        ];
        if let Some(a) = &self.alignment {
            ensure!(
                a.recurrent.input.input == ALIGN_INPUT
                    && a.recurrent.input.bias.len() == 3 * HIDDEN
                    && a.recurrent.recurrent.input == HIDDEN
                    && a.recurrent.recurrent.bias.len() == 3 * HIDDEN
                    && a.window.input == HIDDEN
                    && a.window.bias.len() == 3 * WINDOWS,
                "alignment shape mismatch"
            );
            layers.extend([&a.recurrent.input, &a.recurrent.recurrent, &a.window]);
        }
        for l in layers {
            ensure!(
                l.input > 0
                    && l.weight.len() == l.input * l.bias.len()
                    && l.weight.iter().chain(&l.bias).all(|x| x.is_finite()),
                "invalid sequence weights"
            );
        }
        ensure!(
            self.encoder.input.input == EMBED
                && self.encoder.recurrent.input == HIDDEN
                && self.encoder.input.bias.len() == 3 * HIDDEN
                && self.encoder.recurrent.bias.len() == 3 * HIDDEN
                && self.decoder.input.input == decoder_input_width(self.global_conditioning)
                && self.decoder.recurrent.input == HIDDEN
                && self.decoder.input.bias.len() == 3 * HIDDEN
                && self.decoder.recurrent.bias.len() == 3 * HIDDEN
                && self.query.input == HIDDEN
                && self.query.bias.len() == HIDDEN
                && self.output.input == HIDDEN * 2 + STYLE
                && self.output.bias.len() == PARAMETERS
                && self.length.input == HIDDEN
                && self.length.bias.len() == 1,
            "sequence shape mismatch"
        );
        Ok(())
    }
    pub fn condition(&self, text: &str) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        ensure!(!text.is_empty(), "empty model condition");
        let conditions = if text.chars().count() == 1 {
            vec![text.to_owned()]
        } else {
            ink_layout::tokenize(text)?
        };
        self.condition_tokens(&conditions)
    }
    pub fn condition_tokens(&self, conditions: &[String]) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        ensure!(!conditions.is_empty(), "empty model condition");
        let mut states = vec![];
        let mut h = vec![0.; HIDDEN];
        for token in conditions {
            let i = self
                .vocabulary
                .binary_search(token)
                .map_err(|_| anyhow::anyhow!("untrained input token {token:?}"))?;
            h = self.encoder.step(&self.embedding[i], &h)?;
            states.push(h.clone());
        }
        let mut mean = vec![0.; HIDDEN];
        for h in &states {
            for (i, v) in h.iter().enumerate() {
                mean[i] += v / states.len() as f32;
            }
        }
        Ok((states, mean))
    }
    pub fn step(
        &self,
        previous: &[f32],
        hidden: &[f32],
        states: &[Vec<f32>],
        mean: &[f32],
        style: &[f32],
        progress: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            hidden.len() == HIDDEN
                || (self.alignment.is_some() && hidden.len() == 2 * HIDDEN + 2 * WINDOWS),
            "invalid recurrent state"
        );
        let mut alignment_state = Vec::new();
        let learned_scores = if let Some(a) = &self.alignment {
            let zero = vec![0.; HIDDEN];
            let ah = if hidden.len() > HIDDEN {
                &hidden[HIDDEN..2 * HIDDEN]
            } else {
                &zero
            };
            let next = a.recurrent.step(
                &[previous, mean, style, &[(states.len() as f32).ln()]].concat(),
                ah,
            )?;
            let parameters = a.window.apply(&next)?;
            let mass = softmax(&parameters[..WINDOWS]);
            let mut centers = vec![0.; WINDOWS];
            let mut compensation = vec![0.; WINDOWS];
            for k in 0..WINDOWS {
                let delta = parameters[2 * WINDOWS + k].clamp(-8., 2.).exp();
                centers[k] = if hidden.len() > HIDDEN {
                    let previous = hidden[2 * HIDDEN + k];
                    let increment = delta - hidden[2 * HIDDEN + WINDOWS + k];
                    let next = previous + increment;
                    compensation[k] = (next - previous) - increment;
                    next
                } else {
                    0.
                };
            }
            let scores = (0..states.len())
                .map(|i| {
                    (0..WINDOWS)
                        .map(|k| {
                            mass[k]
                                * (-parameters[WINDOWS + k].clamp(-2., 4.).exp()
                                    * (centers[k] - i as f32).powi(2))
                                .exp()
                        })
                        .sum::<f32>()
                        .max(1e-30)
                        .ln()
                })
                .collect::<Vec<_>>();
            alignment_state.extend(next);
            alignment_state.extend(centers);
            alignment_state.extend(compensation);
            Some(scores)
        } else {
            None
        };
        let progress = if self.alignment.is_some() {
            alignment_state[HIDDEN..HIDDEN + WINDOWS]
                .iter()
                .sum::<f32>()
                / WINDOWS as f32
                / states.len() as f32
        } else {
            progress
        };
        let units = if self.token_space_attention {
            states.len() as f32
        } else {
            1.
        };
        let local;
        let recurrent_text = if self.local_conditioning {
            let probabilities = softmax(
                learned_scores.as_deref().unwrap_or(
                    &states
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            -self.attention_strength
                                * ((progress - i as f32 / states.len() as f32) * units).powi(2)
                        })
                        .collect::<Vec<_>>(),
                ),
            );
            local = (0..HIDDEN)
                .map(|j| {
                    states
                        .iter()
                        .zip(&probabilities)
                        .map(|(s, p)| s[j] * p)
                        .sum()
                })
                .collect::<Vec<f32>>();
            &local[..]
        } else {
            mean
        };
        let input = if self.global_conditioning {
            [
                previous,
                recurrent_text,
                mean,
                style,
                &[progress, (states.len() as f32).ln()],
            ]
            .concat()
        } else {
            [previous, recurrent_text, style, &[progress]].concat()
        };
        let h = self.decoder.step(&input, &hidden[..HIDDEN])?;
        let q = self.query.apply(&h)?;
        let scores: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                s.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() / (HIDDEN as f32).sqrt()
                    + learned_scores.as_ref().map_or_else(
                        || {
                            -self.attention_strength
                                * ((progress - i as f32 / states.len() as f32) * units).powi(2)
                        },
                        |scores| scores[i],
                    )
            })
            .collect();
        let a = softmax(&scores);
        let mut context = vec![0.; HIDDEN];
        for (s, p) in states.iter().zip(a) {
            for (i, v) in s.iter().enumerate() {
                context[i] += v * p;
            }
        }
        let y = self
            .output
            .apply(&[h.as_slice(), &context, style].concat())?;
        let mut state = h;
        state.extend(alignment_state);
        Ok((state, y))
    }
    pub fn generation_condition(&self, label: &str) -> String {
        label.into()
    }
    pub fn generate(
        &self,
        text: &str,
        seed: u64,
        variation: f32,
        document_seed: u64,
    ) -> Result<Vec<Stroke>> {
        ensure!(
            variation.is_finite() && (0.0..=2.0).contains(&variation),
            "variation must be 0..2"
        );
        let condition = self.generation_condition(text);
        if let Some(g) = &self.glyph_head
            && g.supported.iter().any(|s| s == &condition)
        {
            return g.generate(&condition, seed, variation, document_seed);
        }
        let (states, mean) = self.condition(text)?;
        self.generate_conditioned(states, mean, seed, variation, document_seed)
    }
    /// Joint decoding has no glyph-head shortcut, inferred segmentation or added joins.
    pub fn generate_tokens(
        &self,
        conditions: &[String],
        seed: u64,
        variation: f32,
        document_seed: u64,
    ) -> Result<Vec<Stroke>> {
        ensure!(
            variation.is_finite() && (0.0..=2.0).contains(&variation),
            "variation must be 0..2"
        );
        let (states, mean) = self.condition_tokens(conditions)?;
        self.generate_conditioned(states, mean, seed, variation, document_seed)
    }
    fn generate_conditioned(
        &self,
        states: Vec<Vec<f32>>,
        mean: Vec<f32>,
        seed: u64,
        variation: f32,
        document_seed: u64,
    ) -> Result<Vec<Stroke>> {
        let shared = normal(document_seed, STYLE);
        let local = normal(seed, STYLE);
        let style: Vec<_> = shared
            .iter()
            .zip(local)
            .map(|(a, b)| (a * 0.8 + b * if self.whole_run { 0.6 } else { 0.2 }) * variation)
            .collect();
        let mut h = vec![0.; HIDDEN];
        let mut prev = [0.; FEATURES];
        let mut events = vec![];
        let mut started = false;
        let log_length = self.length.apply(&mean)?[0]
            + if self.length_per_token {
                (states.len() as f32).ln()
            } else {
                0.
            };
        let expected = log_length
            .clamp(0., (self.max_events as f32).ln())
            .exp()
            .max(2.);
        for i in 0..self.max_events {
            let (next, y) = self.step(&prev, &h, &states, &mean, &style, i as f32 / expected)?;
            h = next;
            ensure!(y.iter().all(|x| x.is_finite()), "nonfinite decoder output");
            let mut logits = y[..3].to_vec();
            if !started {
                logits[1] = f32::NEG_INFINITY;
                logits[2] = f32::NEG_INFINITY;
            }
            let kind = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .context("event logits")?
                .0;
            let mut e = [0.; FEATURES];
            if kind == 2 {
                e[END] = 1.;
                events.push(e);
                return ink_corpus::sequence::decode(&events, [0., 0.], 1.);
            }
            let weights = softmax(&y[3..3 + MIXTURES]);
            let noise = normal(seed.wrapping_add(i as u64 * 7919 + 17), 8);
            let mix = if variation == 0. || self.whole_run {
                weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            } else {
                let u = 0.5 * (1. + (noise[7] * 0.7978846).tanh());
                let mut sum = 0.;
                weights
                    .iter()
                    .position(|p| {
                        sum += p;
                        sum >= u
                    })
                    .unwrap_or(MIXTURES - 1)
            };
            let offset = 3 + MIXTURES + mix * 14;
            for j in 0..7 {
                let mu = y[offset + j];
                let sigma = y[offset + 7 + j].clamp(-9., 0.).exp();
                e[j] = if self.whole_run {
                    mu
                } else {
                    mu + noise[j] * sigma * variation * 0.35
                };
            }
            e[if kind == 0 { MOVE } else { CURVE }] = 1.;
            if self.canonical_feedback {
                e = ink_corpus::sequence::canonical_feedback(e);
            } else {
                e[6] = e[6].abs().clamp(0.0005, 0.08);
            }
            started = true;
            prev = e;
            events.push(e);
        }
        anyhow::bail!(
            "neural sequence exceeded {} events without termination",
            self.max_events
        )
    }
}

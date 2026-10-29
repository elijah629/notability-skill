//! Whole-row multiple-instance vector loss. Soft assignments stay inside the model.
use anyhow::{Context, Result, ensure};
use ink_corpus::{Sample, Stroke, bounds, file_hash, load, save};
use ink_neural::{
    Linear,
    compositional::{GlyphDecoder, MODES, OUTPUT, POINTS, SLOTS},
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    time::Instant,
};
use tch::{Device, Kind, Tensor, nn};
pub(super) const RECIPE: &str = include_str!("../../../recipes/compositional-vector-v20.json");
type TopologyEvidence = BTreeMap<String, (String, Vec<(usize, usize)>)>;
struct Row {
    sample: Sample,
    labels: Vec<String>,
    target: Vec<f32>,
    width: f32,
    pen_width: Vec<f32>,
    spaces_before: Vec<f32>,
    strong_anchor: bool,
}
/// Transient spatial attention: delayed dots/crossbars can return to earlier cells.
/// No source array is sorted, segmented, relabeled, or exported as a glyph sample.
fn spatial_groups(row: &Row, initial: &[f32]) -> Option<Vec<Vec<usize>>> {
    let n = row.labels.len();
    let xs: Vec<_> = row
        .target
        .as_chunks::<48>()
        .0
        .iter()
        .map(|s| s.as_chunks::<2>().0.iter().map(|p| p[0]).sum::<f32>() / POINTS as f32)
        .collect();
    let mut centers = initial.to_vec();
    let mut groups = vec![vec![]; n];
    for _ in 0..8 {
        groups.iter_mut().for_each(Vec::clear);
        for (source, x) in xs.iter().enumerate() {
            let token = (0..n)
                .min_by(|a, b| (x - centers[*a]).abs().total_cmp(&(x - centers[*b]).abs()))?;
            groups[token].push(source);
        }
        if groups.iter().any(|g| g.is_empty() || g.len() > SLOTS) {
            return None;
        }
        for (i, group) in groups.iter().enumerate() {
            centers[i] = group.iter().map(|s| xs[*s]).sum::<f32>() / group.len() as f32;
        }
    }
    if centers.windows(2).any(|p| p[0] >= p[1]) {
        return None;
    }
    if n > 1 {
        for (i, group) in groups.iter().enumerate() {
            let pitch = if i == 0 {
                centers[1] - centers[0]
            } else if i + 1 == n {
                centers[i] - centers[i - 1]
            } else {
                (centers[i + 1] - centers[i - 1]) * 0.5
            };
            for s in group {
                let points = &row.target[*s * 48..(*s + 1) * 48];
                let min = points
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| p[0])
                    .fold(f32::INFINITY, f32::min);
                let max = points
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| p[0])
                    .fold(f32::NEG_INFINITY, f32::max);
                if max - min > pitch * 1.5 {
                    return None;
                }
            }
        }
    }
    Some(groups)
}
fn points(stroke: &Stroke) -> Vec<[f32; 2]> {
    // Arc-length resampling retains whole stroke order without character splitting.
    let mut dense = vec![stroke.points[0]];
    for p in stroke.points.windows(4).step_by(3) {
        for j in 1..=16 {
            let t = j as f32 / 16.;
            let q = 1. - t;
            dense.push(std::array::from_fn(|a| {
                q * q * q * p[0][a]
                    + 3. * q * q * t * p[1][a]
                    + 3. * q * t * t * p[2][a]
                    + t * t * t * p[3][a]
            }));
        }
    }
    let mut lengths = vec![0_f32];
    for pair in dense.windows(2) {
        lengths.push(
            lengths.last().unwrap() + (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]),
        );
    }
    let total = *lengths.last().unwrap();
    (0..POINTS)
        .map(|j| {
            let d = total * j as f32 / (POINTS - 1) as f32;
            let k = lengths
                .partition_point(|v| *v < d)
                .clamp(1, dense.len().saturating_sub(1).max(1));
            if dense.len() == 1 {
                return dense[0];
            }
            let t = (d - lengths[k - 1]) / (lengths[k] - lengths[k - 1]).max(1e-8);
            std::array::from_fn(|a| dense[k - 1][a] * (1. - t) + dense[k][a] * t)
        })
        .collect()
}

fn eligible(s: &Sample) -> Result<Option<Row>> {
    let tokens = ink_layout::tokenize(&s.label)?;
    if tokens
        .iter()
        .any(|t| t.starts_with('<') && t.ends_with('>') && t.len() > 1 && t != "<space>")
    {
        return Ok(None);
    }
    let mut labels = vec![];
    let mut spaces_before = vec![];
    let mut spaces = 0.;
    for token in tokens {
        if token == "<space>" {
            spaces += 1.;
        } else {
            labels.push(token);
            spaces_before.push(spaces);
        }
    }
    if labels.is_empty() {
        return Ok(None);
    }
    // Ordinary connected writing belongs to the whole-sequence model. The glyph
    // auxiliary must not turn spatial guesses into disjoint character shapes.
    if labels.len() > 1 && !spaces_before.windows(2).all(|p| p[1] > p[0]) {
        return Ok(None);
    }
    if s.strokes.len() > labels.len() * SLOTS {
        return Ok(None);
    }
    let b = bounds(s);
    let mut h = (b[3] - b[1]).max(0.01);
    if labels.len() == 1 {
        let (a, z) = ink_layout::font_metrics(&labels[0]);
        let row_extent = a.max(0.) + z.max(0.);
        let glyph_height = (a + z).max(0.01);
        if [
            "-", "=", "+", "*", "\\times", "\\div", "\\pm", "\\leq", "\\geq", "\\neq", "\\approx",
            "\\sim", "\\propto", "\\in", "\\circ",
        ]
        .contains(&labels[0].as_str())
        {
            h = (b[2] - b[0]).max(b[3] - b[1]).max(0.01) / 0.8 * row_extent.max(0.1);
        } else if labels[0] == "." {
            let pen = s.strokes.iter().map(|s| s.width).fold(0., f32::max);
            h = (b[2] - b[0]).max(b[3] - b[1]).max(pen) / 0.06 * row_extent.max(0.01);
        } else {
            h *= row_extent / glyph_height;
        }
    }

    let width = (b[2] - b[0]) / h;
    let target = s
        .strokes
        .iter()
        .flat_map(points)
        .flat_map(|p| [(p[0] - b[0]) / h, (p[1] - b[1]) / h])
        .collect();
    let pen_width = s.strokes.iter().map(|s| s.width / h).collect();
    let strong_anchor = ink_corpus::label_geometry_warnings(std::slice::from_ref(s)).is_empty()
        && !(labels.len() == 1 && labels[0] == "9");
    Ok(Some(Row {
        sample: s.clone(),
        labels,
        target,
        width,
        pen_width,
        spaces_before,
        strong_anchor,
    }))
}
fn compose_ink_x(predicted: &Tensor, advances: &Tensor, extent: f32, centers: &Tensor) -> Tensor {
    (predicted.select(-1, 0) - 0.5) * advances.reshape([-1, 1, 1]) / extent as f64 + centers
}
/// Reuse complete short contexts for labels lacking singleton/practice anchors.
/// These are sampling choices, never inferred character boundaries or labels.
fn balanced_contexts(rows: &[Row]) -> std::collections::BTreeMap<String, Vec<usize>> {
    let mut candidates = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (i, row) in rows.iter().enumerate().filter(|(_, r)| r.strong_anchor) {
        for label in row.labels.iter().collect::<BTreeSet<_>>() {
            candidates.entry(label.clone()).or_default().push(i);
        }
    }
    for choices in candidates.values_mut() {
        choices.sort_by_key(|i| {
            let r = &rows[*i];
            (
                r.labels.iter().collect::<BTreeSet<_>>().len(),
                r.labels.len(),
                r.sample.strokes.len(),
                *i,
            )
        });
        choices.truncate(8);
    }
    candidates
}
fn mix_contexts(rows: &[Row], by_label: &mut std::collections::BTreeMap<String, Vec<usize>>) {
    for (label, contexts) in balanced_contexts(rows) {
        let choices = by_label.entry(label).or_default();
        if choices.is_empty() {
            for i in contexts {
                if !choices.contains(&i) {
                    choices.push(i);
                }
            }
        }
    }
}
fn prefix_occupancy(counts: &Tensor) -> Tensor {
    ((counts - Tensor::arange(SLOTS as i64, (Kind::Float, counts.device())).unsqueeze(0) - 0.5)
        * 20.)
        .sigmoid()
        .reshape([-1])
        .clamp(0.0001, 0.9999)
}
fn curvature_residual(prediction: &Tensor, target: &Tensor) -> Tensor {
    let difference = prediction - target;
    let curvature = difference.narrow(2, 2, (POINTS - 2) as i64)
        - difference.narrow(2, 1, (POINTS - 2) as i64) * 2.
        + difference.narrow(2, 0, (POINTS - 2) as i64);
    curvature
        .square()
        .mean_dim(&[-1_i64, -2][..], false, Kind::Float)
}
fn point_set_distance(a: &Tensor, b: &Tensor) -> Tensor {
    let squared = (a.unsqueeze(-2) - b.unsqueeze(-3))
        .square()
        .sum_dim_intlist(&[-1_i64][..], false, Kind::Float);
    let forward = squared.min_dim(-1, false).0;
    let backward = squared.min_dim(-2, false).0;
    // Mean distance alone can sacrifice a short stem to fit the larger loop.
    forward.mean_dim(&[-1_i64][..], false, Kind::Float)
        + backward.mean_dim(&[-1_i64][..], false, Kind::Float)
        + (forward.max_dim(-1, false).0 + backward.max_dim(-1, false).0) * 0.05
}
fn shape_loss(prediction: &Tensor, target: &Tensor, matching: &Tensor, masks: &Tensor) -> Tensor {
    let count = target.size()[0];
    let k = 4.min(prediction.size()[0]);
    let indices = matching.transpose(0, 1).topk(k, -1, false, true).1;
    let chosen = prediction
        .reshape([-1, POINTS as i64, 2])
        .index_select(0, &indices.reshape([-1]))
        .reshape([count, k, POINTS as i64, 2]);
    let reference = target.reshape([count, 1, POINTS as i64, 2]);
    let distance = point_set_distance(&chosen, &reference);
    let probability = masks
        .index_select(0, &indices.reshape([-1]))
        .reshape([count, k]);
    -(-distance / 0.02 + probability.log())
        .logsumexp(-1, false)
        .mean(Kind::Float)
        * 0.02
}
struct Head {
    store: nn::VarStore,
    embedding: nn::Embedding,
    advances: nn::Embedding,
    residual: nn::Embedding,
    a: nn::Linear,
    b: nn::Linear,
    out: nn::Linear,
    evidence: RefCell<TopologyEvidence>,
    strong_labels: BTreeSet<String>,
}
fn restore_topology_evidence(net: &Head, folder: &Path, state: &Value) -> Result<()> {
    let path = folder.join("glyph-topology-evidence.json");
    if !state["topology_evidence_sha256"].is_null() {
        ensure!(
            state["topology_evidence_sha256"] == file_hash(&path)?,
            "topology checkpoint integrity"
        );
        *net.evidence.borrow_mut() = serde_json::from_slice(&fs::read(path)?)?;
    }
    Ok(())
}
impl Head {
    fn initialize_topologies(&self, size: usize) -> Result<()> {
        let tokens = Tensor::arange(size as i64, (Kind::Int64, self.store.device()));
        let z = Tensor::zeros([size as i64, 16], (Kind::Float, self.store.device()));
        let y = Vec::<Vec<f32>>::try_from(tch::no_grad(|| {
            self.forward_modes(&tokens, &z)
                .reshape([size as i64, (MODES * OUTPUT) as i64])
                .to_device(Device::Cpu)
        }))?;
        tch::no_grad(|| -> Result<()> {
            for (token, row) in y.iter().enumerate() {
                for mode in 0..MODES {
                    let desired = (mode / 2 + 1) as f32;
                    let mut low = -24_f32;
                    let mut high = 24_f32;
                    for _ in 0..40 {
                        let shift = (low + high) * 0.5;
                        let sum: f32 = (0..SLOTS)
                            .map(|s| {
                                let p = row[mode * OUTPUT + s * 50 + 49].clamp(1e-6, 1. - 1e-6);
                                1. / (1. + (-(p / (1. - p)).ln() - shift).exp())
                            })
                            .sum();
                        if sum < desired {
                            low = shift;
                        } else {
                            high = shift;
                        }
                    }
                    let noise = ink_neural::normal(419 + token as u64 * 137 + mode as u64, OUTPUT);
                    let delta: Vec<_> = (0..OUTPUT)
                        .map(|j| {
                            if j % 50 == 49 {
                                (low + high) * 0.5
                            } else if j % 50 < 48 && mode % 2 == 1 {
                                noise[j] * 0.02
                            } else {
                                0.
                            }
                        })
                        .collect();
                    let mut bias = self.residual.ws.get(token as i64).narrow(
                        0,
                        (mode * OUTPUT) as i64,
                        OUTPUT as i64,
                    );
                    let _ =
                        bias.f_add_(&Tensor::from_slice(&delta).to_device(self.store.device()))?;
                }
            }
            Ok(())
        })
    }
    fn new(device: Device, vocab: usize) -> Self {
        let store = nn::VarStore::new(device);
        let p = &store.root();
        Self {
            embedding: nn::embedding(p / "embedding", vocab as i64, 32, Default::default()),
            advances: nn::embedding(
                p / "advances",
                vocab as i64,
                MODES as i64,
                nn::EmbeddingConfig {
                    ws_init: nn::Init::Const(0.),
                    ..Default::default()
                },
            ),
            residual: nn::embedding(
                p / "residual",
                vocab as i64,
                (OUTPUT * MODES) as i64,
                nn::EmbeddingConfig {
                    ws_init: nn::Init::Const(0.),
                    ..Default::default()
                },
            ),
            a: nn::linear(p / "a", 48, 128, Default::default()),
            b: nn::linear(p / "b", 128, 256, Default::default()),
            out: nn::linear(p / "out", 256, OUTPUT as i64, Default::default()),
            evidence: RefCell::new(BTreeMap::new()),
            strong_labels: BTreeSet::new(),
            store,
        }
    }
    fn forward(&self, tokens: &Tensor, z: &Tensor) -> Tensor {
        self.forward_modes(tokens, z).select(1, 0)
    }
    fn forward_modes(&self, tokens: &Tensor, z: &Tensor) -> Tensor {
        let logits = Tensor::cat(&[tokens.apply(&self.embedding), z.shallow_clone()], -1)
            .apply(&self.a)
            .relu()
            .apply(&self.b)
            .relu()
            .apply(&self.out);
        (logits.unsqueeze(1)
            + tokens
                .apply(&self.residual)
                .reshape([-1, MODES as i64, OUTPUT as i64]))
        .sigmoid()
        .reshape([-1, MODES as i64, SLOTS as i64, 50])
    }
    fn loss(&self, row: &Row, vocab: &[String], step: u64) -> Result<Tensor> {
        if let Some(loss) = self.canonical_loss(row, vocab, step)? {
            return Ok(loss);
        }
        self.weak_loss(row, vocab, step).map(|loss| loss * 0.1)
    }
    fn preferred_modes(&self, size: usize) -> Vec<usize> {
        let mut sources = vec![vec![BTreeSet::new(); MODES]; size];
        let evidence = self.evidence.borrow();
        for (source, choices) in evidence.values() {
            for &(token, mode) in choices {
                sources[token][mode].insert(source);
            }
        }
        sources
            .iter()
            .map(|modes| {
                (0..MODES)
                    .max_by_key(|m| (modes[*m].len(), std::cmp::Reverse(*m)))
                    .unwrap()
            })
            .collect()
    }
    fn canonical_loss(&self, row: &Row, vocab: &[String], step: u64) -> Result<Option<Tensor>> {
        let n = row.labels.len();
        let m = row.sample.strokes.len();
        if m < n || !row.strong_anchor {
            return Ok(None);
        }
        let trusted = n == 1 || row.spaces_before.windows(2).all(|p| p[1] > p[0]);
        let supervised: Vec<_> = row
            .labels
            .iter()
            .map(|label| {
                (trusted || !self.strong_labels.contains(label))
                    && super::style::latin_anchor_allowed(&row.labels, label)
            })
            .collect();
        if !trusted && (n > 8 || supervised.iter().all(|v| !*v)) {
            return Ok(None);
        }
        let device = self.store.device();
        let tokens: Vec<i64> = row
            .labels
            .iter()
            .map(|label| {
                vocab
                    .binary_search(label)
                    .map(|i| i as i64)
                    .map_err(|_| anyhow::anyhow!("missing token"))
            })
            .collect::<Result<_>>()?;
        let noise = ink_neural::normal(419 + step, 16);
        let z = Tensor::from_slice(&noise)
            .to_device(device)
            .unsqueeze(0)
            .expand([n as i64, -1], true)
            * if step.is_multiple_of(2) { 0. } else { 0.4 };
        let token_tensor = Tensor::from_slice(&tokens).to_device(device);
        let y = self.forward_modes(&token_tensor, &z);
        let raw_xy =
            y.narrow(-1, 0, 48)
                .reshape([n as i64, MODES as i64, SLOTS as i64, POINTS as i64, 2])
                * 1.4
                - 0.2;
        let advances = token_tensor.apply(&self.advances).softplus() + 0.2;
        let mean_advances = advances.mean_dim(&[1_i64][..], true, Kind::Float);
        let spaces = Tensor::from_slice(&row.spaces_before)
            .to_device(device)
            .unsqueeze(-1)
            * mean_advances.mean(Kind::Float)
            * 0.5;
        let total = mean_advances.sum(Kind::Float) + spaces.max();
        let centers = ((mean_advances.cumsum(0, Kind::Float) - &mean_advances * 0.5 + spaces)
            / &total
            * row.width as f64)
            .reshape([n as i64, 1, 1, 1]);
        let separated = n > 1 && row.spaces_before.windows(2).all(|p| p[1] > p[0]);
        let vertical: Vec<_> = row
            .labels
            .iter()
            .map(|label| ink_layout::font_metrics(label))
            .collect();
        let above = vertical.iter().map(|(a, _)| *a).fold(0., f32::max);
        let extent = (above + vertical.iter().map(|(_, b)| *b).fold(0., f32::max)).max(0.1);
        let centers = if separated {
            (Tensor::arange(n as i64, (Kind::Float, device)) + 0.5).reshape([n as i64, 1, 1, 1])
                * (row.width as f64 / n as f64)
        } else {
            centers
        };
        let cpu_centers = Vec::<f32>::try_from(centers.to_device(Device::Cpu).reshape([-1]))?;
        let px = (raw_xy.select(-1, 0) - 0.5) * advances.reshape([n as i64, MODES as i64, 1, 1])
            / extent as f64
            + centers;
        let heights: Vec<_> = vertical
            .iter()
            .map(|(a, b)| if separated { 1. } else { (a + b) / extent })
            .collect();
        let offsets: Vec<_> = vertical
            .iter()
            .map(|(a, _)| if separated { 0. } else { (above - a) / extent })
            .collect();
        let py = raw_xy.select(-1, 1)
            * Tensor::from_slice(&heights)
                .to_device(device)
                .reshape([n as i64, 1, 1, 1])
            + Tensor::from_slice(&offsets)
                .to_device(device)
                .reshape([n as i64, 1, 1, 1]);
        let prediction = Tensor::stack(&[px, py], -1).reshape([-1, 48]);
        let target = Tensor::from_slice(&row.target)
            .to_device(device)
            .reshape([m as i64, 48]);
        let forward = (prediction.unsqueeze(1) - target.unsqueeze(0))
            .square()
            .mean_dim(&[-1_i64][..], false, Kind::Float);
        let cost = forward;
        let counts = y
            .select(-1, 49)
            .sum_dim_intlist(&[-1_i64][..], false, Kind::Float);
        let cpu_cost = Vec::<f32>::try_from(cost.detach().to_device(Device::Cpu).reshape([-1]))?;
        let cpu_counts =
            Vec::<f32>::try_from(counts.detach().to_device(Device::Cpu).reshape([-1]))?;
        let Some(groups) = spatial_groups(row, &cpu_centers) else {
            return Ok(None);
        };
        let modes: Vec<_> = groups
            .iter()
            .enumerate()
            .map(|(i, group)| {
                (0..MODES)
                    .min_by(|a, b| {
                        let score = |mode: usize| {
                            group
                                .iter()
                                .enumerate()
                                .map(|(slot, s)| {
                                    cpu_cost[((i * MODES + mode) * SLOTS + slot) * m + s]
                                })
                                .sum::<f32>()
                                / group.len() as f32
                                + 0.2 * (cpu_counts[i * MODES + mode] - group.len() as f32).powi(2)
                        };
                        score(*a).total_cmp(&score(*b))
                    })
                    .unwrap()
            })
            .collect();
        self.evidence.borrow_mut().insert(
            row.sample.sample_id.clone(),
            (
                row.sample.source_sha256.clone(),
                tokens
                    .iter()
                    .zip(&modes)
                    .zip(&supervised)
                    .filter(|(_, active)| **active)
                    .map(|((t, mode), _)| (*t as usize, *mode))
                    .collect(),
            ),
        );
        let indices: Vec<_> = modes
            .iter()
            .enumerate()
            .map(|(i, mode)| (i * MODES + mode) as i64)
            .collect();
        let indices = Tensor::from_slice(&indices).to_device(device);
        let selected = raw_xy
            .reshape([-1, SLOTS as i64, POINTS as i64, 2])
            .index_select(0, &indices);
        let selected_y = y.reshape([-1, SLOTS as i64, 50]).index_select(0, &indices);
        let selected_advances = advances.reshape([-1]).index_select(0, &indices);
        let mut canonical = vec![0_f32; n * SLOTS * POINTS * 2];
        let mut active = vec![0_f32; n * SLOTS];
        let mut closed = active.clone();
        let mut widths = vec![];
        let mut lengths = vec![];
        for (i, group) in groups.iter().enumerate() {
            let slice: Vec<_> = group
                .iter()
                .flat_map(|s| row.target[*s * 48..(*s + 1) * 48].iter().copied())
                .collect();
            let mut bounds = [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ];
            for point in slice.as_chunks::<2>().0 {
                bounds[0] = bounds[0].min(point[0]);
                bounds[1] = bounds[1].min(point[1]);
                bounds[2] = bounds[2].max(point[0]);
                bounds[3] = bounds[3].max(point[1]);
            }
            let w = (bounds[2] - bounds[0]).max(0.001);
            let h = (bounds[3] - bounds[1]).max(0.001);
            let (a, b) = vertical[i];
            let em = (a + b).max(0.05);
            let operator = [
                "-", "=", "+", "*", r"\times", r"\div", r"\pm", r"\leq", r"\geq", r"\neq",
                r"\approx", r"\sim", r"\propto", r"\in", r"\circ",
            ]
            .contains(&row.labels[i].as_str());
            let scale = if operator { 0.8 / w.max(h) } else { em / h };
            let width = (w * scale).clamp(0.2, 4.);
            widths.push(width);
            lengths.push(group.len() as f32);
            if !supervised[i] {
                continue;
            }
            for k in 0..group.len() {
                active[i * SLOTS + k] = 1.;
                let points = &slice[k * 48..(k + 1) * 48];
                let dx = points[46] - points[0];
                let dy = points[47] - points[1];
                let chord = dx.hypot(dy);
                let p = points.as_chunks::<2>().0;
                let sw = p.iter().map(|q| q[0]).fold(f32::NEG_INFINITY, f32::max)
                    - p.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min);
                let sh = p.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max)
                    - p.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min);
                closed[i * SLOTS + k] = if sw.hypot(sh) > 0.05 && chord / sw.hypot(sh) < 0.15 {
                    1.
                } else {
                    0.
                };
                for j in 0..POINTS {
                    canonical[(i * SLOTS + k) * 48 + j * 2] =
                        (points[j * 2] - (bounds[0] + bounds[2]) * 0.5) * scale / width + 0.5;
                    canonical[(i * SLOTS + k) * 48 + j * 2 + 1] =
                        (points[j * 2 + 1] - (bounds[1] + bounds[3]) * 0.5) * scale / em + 0.5;
                }
            }
        }
        let target = Tensor::from_slice(&canonical).to_device(device).reshape([
            n as i64,
            SLOTS as i64,
            POINTS as i64,
            2,
        ]);
        let active = Tensor::from_slice(&active)
            .to_device(device)
            .reshape([n as i64, SLOTS as i64]);
        let closed = Tensor::from_slice(&closed)
            .to_device(device)
            .reshape([n as i64, SLOTS as i64]);
        let token_mask = Tensor::from_slice(
            &supervised
                .iter()
                .map(|v| if *v { 1_f32 } else { 0. })
                .collect::<Vec<_>>(),
        )
        .to_device(device);
        let token_total = token_mask.sum(Kind::Float).clamp_min(1.);
        let stroke_total = active.sum(Kind::Float).clamp_min(1.);
        let forward =
            (&selected - &target)
                .square()
                .mean_dim(&[-1_i64, -2][..], false, Kind::Float);
        let geometry = (forward * &active).sum(Kind::Float) / &stroke_total;
        let smooth =
            (curvature_residual(&selected, &target) * &active).sum(Kind::Float) / &stroke_total;
        let closure = ((selected.select(2, 0) - selected.select(2, (POINTS - 1) as i64))
            .square()
            .mean_dim(&[-1_i64][..], false, Kind::Float)
            * closed)
            .sum(Kind::Float)
            / &stroke_total;
        let raw_count =
            selected_y
                .select(-1, 49)
                .sum_dim_intlist(&[-1_i64][..], false, Kind::Float);
        let count = (&raw_count - Tensor::from_slice(&lengths).to_device(device)).square();
        let count = (count * &token_mask).sum(Kind::Float) / &token_total;
        let masks = prefix_occupancy(&raw_count.unsqueeze(-1)).reshape([n as i64, SLOTS as i64]);
        let one = Tensor::ones_like(&active);
        let occupancy = -((&active * masks.log() + (&one - &active) * (&one - masks).log())
            .mean_dim(&[-1_i64][..], false, Kind::Float)
            * &token_mask)
            .sum(Kind::Float)
            / &token_total;
        let width = (selected_advances.log() - Tensor::from_slice(&widths).to_device(device).log())
            .square();
        let width = (width * &token_mask).sum(Kind::Float) / &token_total;
        Ok(Some(
            geometry * 4.
                + smooth * 10.
                + closure * 5.
                + count * 0.2
                + occupancy * 0.1
                + width * 0.2,
        ))
    }
    fn weak_loss(&self, row: &Row, vocab: &[String], step: u64) -> Result<Tensor> {
        let device = self.store.device();
        let tokens: Vec<_> = row
            .labels
            .iter()
            .map(|l| {
                vocab
                    .binary_search(l)
                    .map(|i| i as i64)
                    .map_err(|_| anyhow::anyhow!("glyph token missing"))
            })
            .collect::<Result<_>>()?;
        let n = tokens.len() as i64;
        let noise = ink_neural::normal(419 + step, 16);
        let z = Tensor::from_slice(&noise)
            .to_device(device)
            .unsqueeze(0)
            .expand([n, -1], true);
        let token_tensor = Tensor::from_slice(&tokens).to_device(device);
        let y = self.forward(&token_tensor, &z);
        let predicted = y
            .narrow(-1, 0, 48)
            .reshape([n, SLOTS as i64, POINTS as i64, 2])
            * 1.4
            - 0.2;
        let advances = token_tensor
            .apply(&self.advances)
            .narrow(1, 0, 1)
            .softplus()
            + 0.2;
        let mean_advance = advances.mean(Kind::Float);
        let spaces = Tensor::from_slice(&row.spaces_before)
            .to_device(device)
            .unsqueeze(-1)
            * &mean_advance
            * 0.5;
        let total = advances.sum(Kind::Float) + spaces.max();
        let centers = ((advances.cumsum(0, Kind::Float) - &advances + spaces) / &total
            * row.width as f64)
            .reshape([n, 1, 1]);
        let stride = (&advances / &total * row.width as f64).reshape([n, 1, 1]);
        let separated = n > 1 && row.spaces_before.windows(2).all(|p| p[1] > p[0]);
        let vertical: Vec<_> = row
            .labels
            .iter()
            .map(|label| ink_layout::font_metrics(label))
            .collect();
        let above = vertical.iter().map(|(a, _)| *a).fold(0., f32::max);
        let below = vertical.iter().map(|(_, b)| *b).fold(0., f32::max);
        let extent = (above + below).max(0.1);
        let glyph_centers = if separated {
            (Tensor::arange(n, (Kind::Float, device)) + 0.5).reshape([n, 1, 1])
                * (row.width as f64 / n as f64)
        } else {
            centers + &stride * 0.5
        };
        let px = compose_ink_x(&predicted, &advances, extent, &glyph_centers);
        let heights: Vec<_> = vertical.iter().map(|(a, b)| (a + b) / extent).collect();
        let offsets: Vec<_> = vertical.iter().map(|(a, _)| (above - a) / extent).collect();
        let py = predicted.select(-1, 1)
            * Tensor::from_slice(&heights)
                .to_device(device)
                .reshape([n, 1, 1])
            + Tensor::from_slice(&offsets)
                .to_device(device)
                .reshape([n, 1, 1]);
        let prediction = Tensor::stack(&[px, py], -1).reshape([-1, 48]);
        let raw = Tensor::from_slice(&row.target)
            .reshape([-1, POINTS as i64, 2])
            .to_device(device);
        let sx = 1. + noise[0] as f64 * 0.025;
        let sy = 1. + noise[1] as f64 * 0.025;
        let slant = noise[2] as f64 * 0.025;
        let target = Tensor::stack(
            &[
                raw.select(-1, 0) * sx + raw.select(-1, 1) * slant,
                raw.select(-1, 1) * sy,
            ],
            -1,
        )
        .reshape([-1, 48]);
        let ordered_cost = (prediction
            .square()
            .mean_dim(&[-1_i64][..], true, Kind::Float)
            + target
                .square()
                .mean_dim(&[-1_i64][..], false, Kind::Float)
                .unsqueeze(0)
            - prediction.matmul(&target.transpose(0, 1)) * (2. / 48.))
            .clamp_min(0.);
        let reversed = target
            .reshape([-1, POINTS as i64, 2])
            .flip([1])
            .reshape([-1, 48]);
        let reversed_cost = (prediction.unsqueeze(1) - reversed.unsqueeze(0))
            .square()
            .mean_dim(&[-1_i64][..], false, Kind::Float);
        let cost = ordered_cost.minimum(&reversed_cost);
        let counts = y
            .select(-1, 49)
            .sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        let masks = prefix_occupancy(&counts);
        let temperature = 0.015;
        let matching = &cost / temperature - masks.log().unsqueeze(1);
        let shape = shape_loss(&prediction, &target, &matching, &masks);
        let paths = prediction.reshape([-1, POINTS as i64, 2]);
        let curvature = paths.narrow(1, 2, (POINTS - 2) as i64)
            - paths.narrow(1, 1, (POINTS - 2) as i64) * 2.
            + paths.narrow(1, 0, (POINTS - 2) as i64);
        let velocity =
            paths.narrow(1, 1, (POINTS - 1) as i64) - paths.narrow(1, 0, (POINTS - 1) as i64);
        let curve_penalty = (curvature
            .square()
            .mean_dim(&[1_i64, 2][..], false, Kind::Float)
            * &masks)
            .sum(Kind::Float)
            / masks.sum(Kind::Float)
            * 5.
            + (velocity
                .square()
                .mean_dim(&[1_i64, 2][..], false, Kind::Float)
                * &masks)
                .sum(Kind::Float)
                / masks.sum(Kind::Float)
                * 0.1;
        let forward = -(-matching).logsumexp(0, false).mean(Kind::Float) * temperature;
        let reverse =
            (&cost.min_dim(1, false).0 * &masks).sum(Kind::Float) / masks.sum(Kind::Float);
        // Balanced entropic transport: one generated stroke cannot explain an entire row.
        let log_mass = (&masks / masks.sum(Kind::Float)).log();
        let kernel = -&cost / 0.01;
        let mut v = Tensor::zeros([target.size()[0]], (Kind::Float, device));
        let mut u = Tensor::zeros_like(&masks);
        for _ in 0..12 {
            u = &log_mass - (&kernel + v.unsqueeze(0)).logsumexp(1, false);
            v = -(target.size()[0] as f64).ln() - (&kernel + u.unsqueeze(1)).logsumexp(0, false);
        }
        let transport =
            ((&kernel + u.unsqueeze(1) + v.unsqueeze(0)).exp() * &cost).sum(Kind::Float);
        let count = ((counts.sum(Kind::Float) - row.sample.strokes.len() as f64)
            / row.labels.len() as f64)
            .square();
        let entropy = (&masks * (1.0_f64 - &masks)).mean(Kind::Float);
        let widths = y.select(-1, 48).reshape([-1]) * 0.08;
        let target_width = Tensor::from_slice(&row.pen_width).to_device(device) * extent as f64;
        let assignments = (-&cost / temperature).softmax(1, Kind::Float);
        let expected_width = assignments
            .matmul(&target_width.unsqueeze(1))
            .squeeze_dim(-1);
        let width_loss = ((widths - expected_width).square() * &masks).mean(Kind::Float);
        if row.labels.len() == 1 && row.strong_anchor {
            let len = row.sample.strokes.len() as i64;
            let xy = (prediction.narrow(0, 0, len) - &target)
                .square()
                .mean(Kind::Float);
            let mask_target = Tensor::arange(SLOTS as i64, (Kind::Int64, device))
                .lt(len)
                .to_kind(Kind::Float);
            let one = Tensor::ones_like(&masks);
            let bce = -(&mask_target * masks.log() + (&one - &mask_target) * (&one - &masks).log())
                .mean(Kind::Float);
            return Ok((xy
                + bce * 0.1
                + &count * 0.2
                + width_loss
                + &shape * 0.2
                + &curve_penalty)
                * 10.);
        }
        Ok(forward
            + reverse
            + transport * 4.
            + count * 0.2
            + entropy * 0.02
            + width_loss
            + shape * 0.2
            + curve_penalty)
    }
    fn export(&self, vocab: &[String]) -> Result<GlyphDecoder> {
        let layer = |l: &nn::Linear| -> Result<Linear> {
            Ok(Linear {
                input: l.ws.size()[1] as usize,
                weight: Vec::<f32>::try_from(l.ws.to_device(Device::Cpu).reshape([-1]))?,
                bias: Vec::<f32>::try_from(l.bs.as_ref().context("bias")?.to_device(Device::Cpu))?,
            })
        };
        let d = GlyphDecoder {
            anchor_head: None,
            anchor_conditions: vec![],
            modes: MODES,
            preferred_modes: self.preferred_modes(vocab.len()),
            canonical_em: true,
            residuals: Vec::<Vec<f32>>::try_from(self.residual.ws.to_device(Device::Cpu))?,
            prefix_masks: true,
            advances: Vec::<f32>::try_from(
                (self.advances.ws.softplus() + 0.2)
                    .to_device(Device::Cpu)
                    .reshape([-1]),
            )?,
            vocabulary: vocab.to_vec(),
            supported: vocab.to_vec(),
            embedding: Vec::<Vec<f32>>::try_from(self.embedding.ws.to_device(Device::Cpu))?,
            layers: vec![layer(&self.a)?, layer(&self.b)?, layer(&self.out)?],
        };
        d.validate()?;
        Ok(d)
    }
}
pub fn train(
    corpus: &Path,
    target: &Path,
    device: Device,
    seconds: f64,
    resume: bool,
) -> Result<Value> {
    let all = load(corpus)?;
    let mut rows = vec![];
    for s in all.iter().filter(|s| s.split == "train") {
        if let Some(r) = eligible(s)? {
            rows.push(r);
        }
    }
    if rows.is_empty() {
        return Ok(
            json!({"status":"skipped","reason":"no supplied isolated/separated rows; complete sequences remain fully supervised","rows":0}),
        );
    }
    let vocab: Vec<_> = rows
        .iter()
        .flat_map(|r| r.labels.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut net = Head::new(device, vocab.len());
    net.strong_labels = rows
        .iter()
        .filter(|row| {
            row.strong_anchor
                && (row.labels.len() == 1 || row.spaces_before.windows(2).all(|p| p[1] > p[0]))
        })
        .flat_map(|row| row.labels.iter().cloned())
        .collect();
    let mut adam = super::sequence::Adam::new();
    let checkpoint = target.join("glyph-checkpoint.safetensors");
    let optimizer = target.join("glyph-optimizer.safetensors");
    let state_path = target.join("glyph-training.json");
    let mut previous_repair_updates = 0_u64;
    let mut initialize_topology = !resume;
    if resume && state_path.exists() {
        let state: Value = serde_json::from_slice(&fs::read(&state_path)?)?;
        ensure!(
            state["corpus_sha256"] == file_hash(corpus)?
                && state["vocabulary"] == json!(vocab)
                && state["recipe"] == serde_json::from_str::<Value>(RECIPE)?,
            "glyph resume data mismatch"
        );
        ensure!(
            state["checkpoint_sha256"] == file_hash(&checkpoint)?
                && state["optimizer_sha256"] == file_hash(&optimizer)?,
            "glyph checkpoint integrity"
        );
        previous_repair_updates = state["repair_updates"].as_u64().unwrap_or(0);
        restore_topology_evidence(&net, target, &state)?;
        net.store.load(&checkpoint)?;
        adam = super::sequence::Adam::restore(
            &optimizer,
            state["updates"].as_u64().context("glyph updates")?,
            &net.store,
        )?;
    }
    let started = Instant::now();
    let mut checkpoint_time = Instant::now();
    let mut loss_value = 0.;
    let initial_step = adam.step;
    if !resume && target.join("parent-model.json").exists() {
        let parent: Value = serde_json::from_slice(&fs::read(target.join("parent-model.json"))?)?;
        if let Some(path) = parent["parent"].as_str() {
            let path = Path::new(path);
            if path.join("glyph-training.json").exists() {
                let state: Value =
                    serde_json::from_slice(&fs::read(path.join("glyph-training.json"))?)?;
                if state["corpus_sha256"] == file_hash(corpus)?
                    && state["recipe"]["mask_mode"] == "prefix-count"
                {
                    initialize_topology = state["recipe"]["modes"].as_u64() != Some(MODES as u64);
                    ensure!(
                        state["checkpoint_sha256"]
                            == file_hash(&path.join("glyph-checkpoint.safetensors"))?,
                        "parent glyph checkpoint mismatch"
                    );
                    let old_vocabulary: Vec<String> =
                        serde_json::from_value(state["vocabulary"].clone())?;
                    let old: std::collections::BTreeMap<_, _> =
                        Tensor::read_safetensors(path.join("glyph-checkpoint.safetensors"))?
                            .into_iter()
                            .collect();
                    tch::no_grad(|| -> Result<()> {
                        for (name, mut weight) in net.store.variables() {
                            if let Some(source) = old.get(&name) {
                                let source = source.to_device(device);
                                if name == "embedding.weight"
                                    || name == "advances.weight"
                                    || name == "residual.weight"
                                {
                                    for (i, label) in vocab.iter().enumerate() {
                                        if let Ok(j) = old_vocabulary.binary_search(label) {
                                            let source_row = source.get(j as i64);
                                            let destination = weight.get(i as i64);
                                            let source_row =
                                                if source_row.size() != destination.size() {
                                                    ensure!(
                                                        destination
                                                            .numel()
                                                            .is_multiple_of(source_row.numel()),
                                                        "incompatible parent mode dimensions"
                                                    );
                                                    source_row.repeat([(destination.numel()
                                                        / source_row.numel())
                                                        as i64])
                                                } else {
                                                    source_row
                                                };
                                            weight.get(i as i64).f_copy_(&source_row)?;
                                        }
                                    }
                                } else {
                                    ensure!(
                                        source.size() == weight.size(),
                                        "parent glyph layer shape mismatch"
                                    );
                                    weight.f_copy_(&source)?;
                                }
                            }
                        }
                        Ok(())
                    })?;
                }
            }
        }
    }
    if initialize_topology {
        net.initialize_topologies(vocab.len())?;
    }
    let anchors: Vec<_> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let mut runs: Vec<(&str, usize)> = vec![];
            for label in &r.labels {
                if let Some(last) = runs.last_mut().filter(|x| x.0 == label) {
                    last.1 += 1;
                } else {
                    runs.push((label, 1));
                }
            }
            if r.strong_anchor
                && (r.labels.len() == 1
                    || (r.labels.len() >= 4
                        && runs.len() <= 4
                        && runs.iter().all(|(_, n)| *n >= 2)))
            {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let mut by_label = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for i in anchors {
        for label in rows[i].labels.iter().collect::<BTreeSet<_>>() {
            by_label.entry(label.clone()).or_default().push(i);
        }
    }
    mix_contexts(&rows, &mut by_label);
    let balanced: Vec<_> = by_label.values().collect();
    let mut order: Vec<_> = (0..rows.len()).collect();
    order.sort_by_key(|i| rows[*i].labels.len());
    let progress = super::progress::Progress::timed("Glyph auxiliary", seconds);
    while started.elapsed().as_secs_f64() < seconds * 0.7 {
        let step = adam.step as usize;
        let cycle = step / 20;
        let i = if step % 20 < 19 && !balanced.is_empty() {
            let k = cycle * 19 + step % 20;
            let choices = balanced[k % balanced.len()];
            choices[(k / balanced.len()) % choices.len()]
        } else {
            order[cycle % order.len()]
        };
        tch::manual_seed((419 + adam.step) as i64);
        let loss = net.loss(&rows[i], &vocab, adam.step)?;
        loss_value = loss.double_value(&[]);
        ensure!(loss_value.is_finite(), "glyph loss nonfinite");
        loss.backward();
        adam.update(&net.store)?;
        progress.update_time(started.elapsed(), format!("{} updates", adam.step));
        if checkpoint_time.elapsed().as_secs() >= 60 {
            persist(
                &net,
                &adam,
                &rows,
                &vocab,
                corpus,
                target,
                started.elapsed().as_secs_f64(),
                loss_value,
                initial_step,
                previous_repair_updates,
            )?;
            checkpoint_time = Instant::now();
        }
    }
    let mut repair_updates = previous_repair_updates;
    let mut repair_rows = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (i, row) in rows.iter().enumerate() {
        for label in row.labels.iter().collect::<BTreeSet<_>>() {
            repair_rows.entry(label.clone()).or_default().push(i);
        }
    }
    for (name, weight) in net.store.variables() {
        let _ = weight.set_requires_grad(name == "residual.weight");
    }
    while started.elapsed().as_secs_f64() < seconds {
        let index = repair_updates as usize % vocab.len();
        let label = &vocab[index];
        let choices = by_label
            .get(label)
            .or_else(|| repair_rows.get(label))
            .context("repair source rows")?;
        let row = &rows[choices[(repair_updates as usize / vocab.len()) % choices.len()]];
        let loss = net.loss(row, &vocab, adam.step + repair_updates)?;
        loss_value = loss.double_value(&[]);
        ensure!(loss_value.is_finite(), "nonfinite conditional refinement");
        loss.backward();
        let mask = Tensor::arange(vocab.len() as i64, (Kind::Int64, device))
            .eq(index as i64)
            .to_kind(Kind::Float)
            .unsqueeze(-1);
        let gradient = net.residual.ws.grad();
        ensure!(
            gradient.isfinite().all().int64_value(&[]) != 0,
            "nonfinite conditional gradient"
        );
        tch::no_grad(|| -> Result<()> {
            let mut weight = net.residual.ws.shallow_clone();
            // Keep relative coordinate gradients; sign-only updates give every
            // near-zero coordinate the same jump and make rare shapes jitter.
            let selected = &gradient * &mask;
            let rms = selected.square().sum(Kind::Float) / OUTPUT as f64;
            let direction = (selected / (rms.sqrt() + 1e-8)).clamp(-3., 3.);
            let _ = weight.f_sub_(&(direction * 0.001))?;
            Ok(())
        })?;
        net.residual.ws.zero_grad();
        repair_updates += 1;
        progress.update_time(
            started.elapsed(),
            format!("refinement · {} updates", repair_updates),
        );
        if checkpoint_time.elapsed().as_secs() >= 60 {
            progress.message("Saving auxiliary checkpoint…");
            persist(
                &net,
                &adam,
                &rows,
                &vocab,
                corpus,
                target,
                started.elapsed().as_secs_f64(),
                loss_value,
                initial_step,
                repair_updates,
            )?;
            checkpoint_time = Instant::now();
        }
    }
    for weight in net.store.variables().values() {
        let _ = weight.set_requires_grad(true);
    }
    let mut state = persist(
        &net,
        &adam,
        &rows,
        &vocab,
        corpus,
        target,
        started.elapsed().as_secs_f64(),
        loss_value,
        initial_step,
        repair_updates,
    )?;
    state["repair_updates"] = json!(repair_updates);
    state["repair_optimizer"] =
        json!("stateless per-token RMS-normalized gradient descent, clipped to 3, lr 0.001");
    save(&target.join("glyph-training.json"), &state)?;
    progress.finish("saved");
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn persist(
    net: &Head,
    adam: &super::sequence::Adam,
    rows: &[Row],
    vocab: &[String],
    corpus: &Path,
    target: &Path,
    seconds: f64,
    loss_value: f64,
    initial_step: u64,
    repair_updates: u64,
) -> Result<Value> {
    net.store
        .save(target.join("glyph-checkpoint.safetensors"))?;
    adam.save(&target.join("glyph-optimizer.safetensors"))?;
    let decoder = net.export(vocab)?;
    let tokens = Tensor::arange(vocab.len() as i64, (Kind::Int64, net.store.device()));
    let z = Tensor::zeros([vocab.len() as i64, 16], (Kind::Float, net.store.device()));
    let expected = Vec::<Vec<f32>>::try_from(
        tch::no_grad(|| net.forward_modes(&tokens, &z))
            .reshape([(vocab.len() * MODES) as i64, OUTPUT as i64])
            .to_device(Device::Cpu),
    )?;
    let mut parity = 0_f32;
    for (i, target) in expected.iter().enumerate() {
        let token = i / MODES;
        let mode = i % MODES;
        let mut x = [decoder.embedding[token].as_slice(), &[0.; 16]].concat();
        for (j, l) in decoder.layers.iter().enumerate() {
            x = l.apply(&x)?;
            for (k, v) in x.iter_mut().enumerate() {
                *v = if j == 2 {
                    1. / (1. + (-(*v + decoder.residuals[token][mode * OUTPUT + k])).exp())
                } else {
                    v.max(0.)
                };
            }
        }
        for (a, b) in x.iter().zip(target) {
            parity = parity.max((a - b).abs());
        }
    }
    ensure!(parity < 0.0001, "glyph decoder parity error {parity}");
    save(&target.join("glyph-decoder.json"), &decoder)?;
    let lineage:Vec<_>=rows.iter().map(|r|json!({"sample_id":r.sample.sample_id,"source_sha256":r.sample.source_sha256,"split":"train","label":r.sample.label,"conditions":r.labels,"strong_anchor":r.strong_anchor,"label_geometry_status":if r.strong_anchor{"supplied label; not independently certified"}else{"recorded singleton geometry ambiguity; retained only in whole-row weak supervision"},"supervision":"whole flat-row multiple-instance vector reconstruction; assignments are internal soft attention, not glyph annotations"})).collect();
    save(&target.join("glyph-lineage.json"), &lineage)?;
    save(
        &target.join("glyph-topology-evidence.json"),
        &*net.evidence.borrow(),
    )?;
    let state = json!({"updates":adam.step,"repair_updates":repair_updates,"new_updates":adam.step-initial_step,"seconds":seconds,"rows":rows.len(),"vocabulary":vocab,"recipe":serde_json::from_str::<Value>(RECIPE)?,"corpus_sha256":file_hash(corpus)?,"checkpoint_sha256":file_hash(&target.join("glyph-checkpoint.safetensors"))?,"optimizer_sha256":file_hash(&target.join("glyph-optimizer.safetensors"))?,"loss":loss_value,"rust_decoder_max_error":parity,"topology_evidence_sha256":file_hash(&target.join("glyph-topology-evidence.json"))?});
    save(&target.join("glyph-training.json"), &state)?;
    Ok(state)
}

pub fn snapshot(model: &Path, out: &Path) -> Result<Value> {
    ensure!(!out.exists(), "snapshot requires fresh output directory");
    let mut manifest = super::show(model)?;
    let state: Value = serde_json::from_slice(&fs::read(model.join("glyph-training.json"))?)?;
    ensure!(
        state["recipe"] == serde_json::from_str::<Value>(RECIPE)?,
        "snapshot recipe mismatch"
    );
    let checkpoint = model.join("glyph-checkpoint.safetensors");
    let checkpoint_hash = file_hash(&checkpoint)?;
    ensure!(
        state["checkpoint_sha256"] == checkpoint_hash
            && state["optimizer_sha256"] == file_hash(&model.join("glyph-optimizer.safetensors"))?,
        "checkpoint is changing; retry snapshot after save completes"
    );
    let vocabulary: Vec<String> = serde_json::from_value(state["vocabulary"].clone())?;
    let mut net = Head::new(Device::Cpu, vocabulary.len());
    net.store.load(&checkpoint)?;
    restore_topology_evidence(&net, model, &state)?;
    let glyph = net.export(&vocabulary)?;
    ensure!(
        file_hash(&checkpoint)? == checkpoint_hash,
        "checkpoint changed while snapshotting; retry"
    );
    let mut decoder = ink_neural::sequence::SequenceDecoder::load(&model.join("decoder.json"))?;
    decoder.glyph_head = Some(glyph);
    decoder.validate()?;
    fs::create_dir_all(out)?;
    save(&out.join("decoder.json"), &decoder)?;
    fs::copy(model.join("vocabulary.json"), out.join("vocabulary.json"))?;
    manifest["decoder_sha256"] = json!(file_hash(&out.join("decoder.json"))?);
    manifest
        .as_object_mut()
        .unwrap()
        .remove("glyph_mode_review");
    manifest["approved_for_default"] = json!(false);
    manifest["glyph_conditions"] = json!(vocabulary.len());
    manifest["glyph_training"] = state.clone();
    manifest["glyph_recipe"] = state["recipe"].clone();
    manifest["snapshot_checkpoint_sha256"] = json!(checkpoint_hash);
    save(&out.join("manifest.json"), &manifest)?;
    Ok(
        json!({"snapshot":out,"model":manifest["id"],"glyph_conditions":vocabulary.len(),"updates":state["updates"],"approved_for_default":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topology_checkpoint_restores_mode_selection_and_identical_adam_update() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let vocabulary = vec!["x".into()];
        let sample = Sample {
            sample_id: "topology-fixture".into(),
            label: "x".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: Value::Null,
            strokes: vec![Stroke {
                points: vec![[0., 0.], [0.3, 0.3], [0.6, 0.6], [1., 1.]],
                width: 0.025,
                source_index: 0,
                uuid: "fixture".into(),
            }],
        };
        let row = eligible(&sample).unwrap().unwrap();
        let net = Head::new(Device::Cpu, 1);
        net.initialize_topologies(1).unwrap();
        let mut adam = super::super::sequence::Adam::new();
        net.loss(&row, &vocabulary, adam.step).unwrap().backward();
        adam.update(&net.store).unwrap();
        let directory =
            std::env::temp_dir().join(format!("ink-topology-contract-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        net.store.save(directory.join("head.safetensors")).unwrap();
        adam.save(&directory.join("adam.safetensors")).unwrap();
        save(
            &directory.join("glyph-topology-evidence.json"),
            &*net.evidence.borrow(),
        )
        .unwrap();
        let state = json!({"topology_evidence_sha256":file_hash(&directory.join("glyph-topology-evidence.json")).unwrap()});
        let mut restored = Head::new(Device::Cpu, 1);
        restored
            .store
            .load(directory.join("head.safetensors"))
            .unwrap();
        restore_topology_evidence(&restored, &directory, &state).unwrap();
        assert_eq!(net.preferred_modes(1), restored.preferred_modes(1));
        let mut restored_adam = super::super::sequence::Adam::restore(
            &directory.join("adam.safetensors"),
            adam.step,
            &restored.store,
        )
        .unwrap();
        net.loss(&row, &vocabulary, adam.step).unwrap().backward();
        adam.update(&net.store).unwrap();
        restored
            .loss(&row, &vocabulary, restored_adam.step)
            .unwrap()
            .backward();
        restored_adam.update(&restored.store).unwrap();
        let a = net.forward_modes(
            &Tensor::from_slice(&[0_i64]),
            &Tensor::zeros([1, 16], (Kind::Float, Device::Cpu)),
        );
        let b = restored.forward_modes(
            &Tensor::from_slice(&[0_i64]),
            &Tensor::zeros([1, 16], (Kind::Float, Device::Cpu)),
        );
        assert_eq!((a - b).abs().max().double_value(&[]), 0.);
        fs::write(directory.join("glyph-topology-evidence.json"), "{}").unwrap();
        assert!(restore_topology_evidence(&restored, &directory, &state).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn whole_row_loss_updates_and_rust_head_matches_native() {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let sample = Sample {
            sample_id: "whole-row".into(),
            label: r"\text{x x x x}".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: Value::Null,
            strokes: (0..4)
                .map(|i| Stroke {
                    points: vec![
                        [i as f32, 0.],
                        [i as f32 + 0.1, 0.2],
                        [i as f32 + 0.2, 0.3],
                        [i as f32 + 0.4, 1.],
                    ],
                    width: 0.02,
                    source_index: i,
                    uuid: i.to_string(),
                })
                .collect(),
        };
        let row = eligible(&sample).unwrap().unwrap();
        assert_eq!(row.labels, vec!["x"; 4]);
        assert_eq!(row.sample.strokes.len(), 4);
        let vocabulary = vec!["x".into()];
        let net = Head::new(Device::Cpu, 1);
        let mut adam = super::super::sequence::Adam::new();
        let before = net.export(&vocabulary).unwrap();
        let loss = net.loss(&row, &vocabulary, 0).unwrap();
        loss.backward();
        adam.update(&net.store).unwrap();
        let after = net.export(&vocabulary).unwrap();
        assert_ne!(before.layers[2].weight, after.layers[2].weight);
        let seed = 419;
        let shared = ink_neural::normal(seed, 16);
        let local = ink_neural::normal(seed, 16);
        let z: Vec<_> = shared
            .iter()
            .zip(local)
            .map(|(a, b)| a * 0.6 + b * 0.4)
            .collect();
        let expected = net
            .forward(
                &Tensor::from_slice(&[0_i64]),
                &Tensor::from_slice(&z).unsqueeze(0),
            )
            .reshape([-1]);
        let expected = Vec::<f32>::try_from(expected).unwrap();
        let mut x = [after.embedding[0].as_slice(), z.as_slice()].concat();
        for (i, l) in after.layers.iter().enumerate() {
            x = l.apply(&x).unwrap();
            for (k, v) in x.iter_mut().enumerate() {
                *v = if i == 2 {
                    1. / (1. + (-(*v + after.residuals[0][k])).exp())
                } else {
                    v.max(0.)
                };
            }
        }
        let error = x
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(error < 1e-6);
        let a = after.generate("x", 419, 1., 419).unwrap();
        let b = after.generate("x", 419, 1., 419).unwrap();
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        assert_ne!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&after.generate("x", 420, 1., 419).unwrap()).unwrap()
        );
        assert!(after.generate("z", 419, 1., 419).is_err());
    }
}

#[cfg(test)]
mod geometry_contracts {
    use super::*;
    #[test]
    fn source_corner_is_not_penalized_for_being_sharp() {
        let mut sharp = vec![];
        let mut rounded = vec![];
        for j in 0..POINTS {
            let t = j as f32 / (POINTS - 1) as f32;
            sharp.extend([t, (t - 0.5).abs() * 2.]);
            rounded.extend([t, (t - 0.5).powi(2) * 4.]);
        }
        let sharp = Tensor::from_slice(&sharp).reshape([1, 1, POINTS as i64, 2]);
        let rounded = Tensor::from_slice(&rounded).reshape([1, 1, POINTS as i64, 2]);
        assert_eq!(curvature_residual(&sharp, &sharp).double_value(&[0, 0]), 0.);
        assert!(curvature_residual(&rounded, &sharp).double_value(&[0, 0]) > 0.0001);
    }
    #[test]
    fn delayed_crossbars_return_to_their_cell_without_reordering_source() {
        let mut row = Row {
            sample: Sample {
                sample_id: "cells".into(),
                label: "tt".into(),
                source_sha256: "a".repeat(64),
                split: "train".into(),
                metadata: Value::Null,
                strokes: vec![],
            },
            labels: vec!["t".into(), "t".into()],
            target: vec![],
            width: 2.,
            pen_width: vec![],
            spaces_before: vec![0., 1.],
            strong_anchor: true,
        };
        for x in [0.5_f32, 1.5, 0.5, 1.5] {
            for j in 0..POINTS {
                row.target.extend([x, j as f32 / POINTS as f32]);
            }
        }
        let original = row.target.clone();
        assert_eq!(
            spatial_groups(&row, &[0.5, 1.5]).unwrap(),
            [vec![0, 2], vec![1, 3]]
        );
        assert_eq!(row.target, original);
        row.target[..48]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
            .for_each(|(j, p)| p[0] = j as f32 * 0.2);
        assert!(spatial_groups(&row, &[0.5, 1.5]).is_none());
    }
    #[test]
    fn topology_initialization_and_distinct_source_selection() {
        tch::set_num_threads(1);
        let net = Head::new(Device::Cpu, 1);
        net.initialize_topologies(1).unwrap();
        let y = net.forward_modes(
            &Tensor::from_slice(&[0_i64]),
            &Tensor::zeros([1, 16], (Kind::Float, Device::Cpu)),
        );
        let counts = Vec::<f32>::try_from(
            y.select(-1, 49)
                .sum_dim_intlist(&[-1_i64][..], false, Kind::Float)
                .reshape([-1]),
        )
        .unwrap();
        for (mode, count) in counts.iter().enumerate() {
            assert!((*count - (mode / 2 + 1) as f32).abs() < 1e-4);
        }
        for i in 0..20 {
            net.evidence.borrow_mut().insert(
                format!("same-source-{i}"),
                ("source-a".into(), vec![(0, 0)]),
            );
        }
        net.evidence
            .borrow_mut()
            .insert("b".into(), ("source-b".into(), vec![(0, 2)]));
        net.evidence
            .borrow_mut()
            .insert("c".into(), ("source-c".into(), vec![(0, 2)]));
        assert_eq!(net.preferred_modes(1), [2]);
        let decoder = net.export(&["x".into()]).unwrap();
        assert_eq!(decoder.generate("x", 419, 0., 419).unwrap().len(), 2);
        let recovered: GlyphDecoder =
            serde_json::from_slice(&serde_json::to_vec(&decoder).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_vec(&decoder.generate("x", 419, 0., 419).unwrap()).unwrap(),
            serde_json::to_vec(&recovered.generate("x", 419, 0., 419).unwrap()).unwrap()
        );
    }
    #[test]
    fn training_does_not_credit_strokes_missing_from_integer_count_inference() {
        let counts = Tensor::from_slice(&[2_f32]).reshape([1, 1]);
        let masks = prefix_occupancy(&counts);
        assert!(masks.double_value(&[0]) > 0.999);
        assert!(masks.double_value(&[1]) > 0.999);
        assert!(masks.double_value(&[2]) < 0.001);
    }
    #[test]
    fn rare_labels_reuse_separated_practice_without_segmenting_words() {
        let sample = |id: &str, label: &str| Sample {
            sample_id: id.into(),
            label: label.into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            strokes: vec![Stroke {
                points: vec![[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
                width: 0.025,
                source_index: 0,
                uuid: id.into(),
            }],
            metadata: Value::Null,
        };
        for label in ["begin", "bebe", "kg"] {
            assert!(
                eligible(&sample("ordinary-word", label)).unwrap().is_none(),
                "ordinary words belong to whole-sequence supervision"
            );
        }
        let rows: Vec<_> = [
            sample("long", "b e g i n"),
            sample("short", "b e b e"),
            sample("rare", "k g"),
        ]
        .iter()
        .map(|s| eligible(s).unwrap().unwrap())
        .collect();
        let contexts = balanced_contexts(&rows);
        assert_eq!(contexts["b"], [1, 0]);
        assert_eq!(contexts["k"], [2]);
        assert_eq!(rows[contexts["b"][0]].sample.label, "b e b e");
        assert_eq!(rows[contexts["b"][0]].target.len(), POINTS * 2);
        assert_eq!(rows[contexts["b"][0]].labels, ["b", "e", "b", "e"]);
        assert!(
            contexts
                .values()
                .flatten()
                .all(|i| rows[*i].sample.split == "train")
        );
        let mut existing = std::collections::BTreeMap::from([("b".into(), vec![0])]);
        mix_contexts(&rows, &mut existing);
        assert!(
            !existing["b"].contains(&1),
            "uncertain word correspondence must not displace a known anchor"
        );
    }
    #[test]
    fn arc_length_ignores_unequal_segment_density() {
        let stroke = Stroke {
            points: vec![
                [0., 0.],
                [0.003, 0.],
                [0.006, 0.],
                [0.01, 0.],
                [0.34, 0.],
                [0.67, 0.],
                [1., 0.],
            ],
            width: 0.02,
            source_index: 0,
            uuid: "test".into(),
        };
        let sampled = points(&stroke);
        for (i, p) in sampled.iter().enumerate() {
            assert!((p[0] - i as f32 / (POINTS - 1) as f32).abs() < 0.002);
        }
    }
    #[test]
    fn shape_distance_preserves_reversed_paths_and_penalizes_collapse() {
        let a = Tensor::from_slice(&[0_f32, 0., 0., 0.5, 0., 1.]).reshape([1, 3, 2]);
        let reversed = a.flip([1]);
        assert_eq!(point_set_distance(&a, &reversed).double_value(&[0]), 0.);
        let collapsed = Tensor::from_slice(&[0_f32, 0.5, 0., 0.5, 0., 0.5]).reshape([1, 3, 2]);
        assert!(point_set_distance(&a, &collapsed).double_value(&[0]) > 0.1);
    }
    #[test]
    fn short_stem_cannot_disappear_in_mean_shape_loss() {
        let mut reference = vec![0_f32; POINTS * 2];
        reference[(POINTS - 1) * 2 + 1] = 1.;
        let reference = Tensor::from_slice(&reference).reshape([1, POINTS as i64, 2]);
        let missing_stem = Tensor::zeros_like(&reference);
        assert!(point_set_distance(&reference, &missing_stem).double_value(&[0]) > 0.08);
        assert_eq!(
            point_set_distance(&reference, &reference.flip([1])).double_value(&[0]),
            0.
        );
    }
    #[test]
    fn row_spacing_translates_ink_without_stretching() {
        let p = Tensor::from_slice(&[0_f32, 0., 0.5, 0.5, 1., 1.]).reshape([1, 1, 3, 2]);
        let advance = Tensor::from_slice(&[0.6_f32]).reshape([1, 1]);
        let center = Tensor::zeros([1, 1, 1], (Kind::Float, Device::Cpu));
        let a = compose_ink_x(&p, &advance, 1., &center);
        let b = compose_ink_x(&p, &advance, 1., &(center + 10.));
        assert!(((a.max() - a.min()).double_value(&[]) - 0.6).abs() < 1e-6);
        assert!(((b.max() - b.min()).double_value(&[]) - 0.6).abs() < 1e-6);
    }
}

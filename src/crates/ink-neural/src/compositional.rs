//! Learned compositional glyph head. Contains weights only, no source trajectories.
use crate::{Linear, normal};
use anyhow::{Context, Result, ensure};
use ink_corpus::Stroke;
use serde::{Deserialize, Serialize};
pub const POINTS: usize = 24;
pub const SLOTS: usize = 12;
pub const OUTPUT: usize = SLOTS * (POINTS * 2 + 2);
pub const MODES: usize = 8;
fn one_mode() -> usize {
    1
}
#[derive(Clone, Serialize, Deserialize)]
pub struct GlyphDecoder {
    #[serde(default)]
    pub anchor_head: Option<Box<GlyphDecoder>>,
    #[serde(default)]
    pub anchor_conditions: Vec<String>,
    #[serde(default = "one_mode")]
    pub modes: usize,
    #[serde(default)]
    pub preferred_modes: Vec<usize>,
    #[serde(default)]
    pub canonical_em: bool,
    #[serde(default)]
    pub residuals: Vec<Vec<f32>>,
    #[serde(default)]
    pub prefix_masks: bool,
    #[serde(default)]
    pub advances: Vec<f32>,
    pub vocabulary: Vec<String>,
    pub embedding: Vec<Vec<f32>>,
    pub layers: Vec<Linear>,
    pub supported: Vec<String>,
}
impl GlyphDecoder {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.anchor_conditions.windows(2).all(|p| p[0] < p[1]),
            "invalid anchor routing"
        );
        if let Some(anchor) = &self.anchor_head {
            ensure!(
                anchor.anchor_head.is_none() && anchor.anchor_conditions.is_empty(),
                "nested anchor expert"
            );
            anchor.validate()?;
            ensure!(
                self.anchor_conditions
                    .iter()
                    .all(|s| self.supported.contains(s) && anchor.supported.contains(s)),
                "unsupported anchor condition"
            );
        } else {
            ensure!(self.anchor_conditions.is_empty(), "anchor weights missing");
        }
        ensure!((1..=MODES).contains(&self.modes), "invalid topology modes");
        ensure!(
            self.preferred_modes.is_empty()
                || (self.preferred_modes.len() == self.vocabulary.len()
                    && self.preferred_modes.iter().all(|m| *m < self.modes)),
            "invalid preferred topology"
        );
        ensure!(
            self.vocabulary.windows(2).all(|w| w[0] < w[1])
                && self
                    .supported
                    .iter()
                    .all(|s| self.vocabulary.binary_search(s).is_ok()),
            "invalid glyph vocabulary"
        );
        ensure!(
            self.advances.is_empty()
                || (self.advances.len() == self.vocabulary.len() * self.modes
                    && self.advances.iter().all(|a| a.is_finite() && *a > 0.)),
            "invalid glyph advances"
        );
        ensure!(
            self.embedding.len() == self.vocabulary.len() && self.layers.len() == 3,
            "glyph head shape"
        );
        ensure!(
            self.residuals.is_empty()
                || (self.residuals.len() == self.vocabulary.len()
                    && self
                        .residuals
                        .iter()
                        .all(|r| r.len() == OUTPUT * self.modes && r.iter().all(|v| v.is_finite()))),
            "invalid conditional biases"
        );
        let mut n = 48;
        for l in &self.layers {
            ensure!(
                l.input == n
                    && l.weight.len() == n * l.bias.len()
                    && l.weight.iter().chain(&l.bias).all(|x| x.is_finite()),
                "glyph head weights"
            );
            n = l.bias.len();
        }
        ensure!(n == OUTPUT, "glyph head output");
        ensure!(
            self.embedding
                .iter()
                .all(|r| r.len() == 32 && r.iter().all(|x| x.is_finite())),
            "glyph embeddings"
        );
        Ok(())
    }
    pub fn generate(
        &self,
        label: &str,
        seed: u64,
        variation: f32,
        document_seed: u64,
    ) -> Result<Vec<Stroke>> {
        if self.anchor_conditions.iter().any(|s| s == label) {
            return self
                .anchor_head
                .as_ref()
                .context("anchor weights missing")?
                .generate(label, seed, variation, document_seed);
        }
        ensure!(
            self.supported.iter().any(|s| s == label),
            "glyph head has no training evidence for {label:?}"
        );
        let index = self
            .vocabulary
            .binary_search(&label.to_owned())
            .map_err(|_| anyhow::anyhow!("unknown glyph {label:?}"))?;
        let mode = self.preferred_modes.get(index).copied().unwrap_or(0);
        let shared = normal(document_seed, 16);
        let local = normal(seed, 16);
        let z: Vec<_> = shared
            .iter()
            .zip(local)
            .map(|(a, b)| (a * 0.6 + b * 0.8) * variation)
            .collect();
        let mut x = [self.embedding[index].as_slice(), z.as_slice()].concat();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.apply(&x)?;
            for (j, v) in x.iter_mut().enumerate() {
                *v = if i == 2 {
                    let bias = self
                        .residuals
                        .get(index)
                        .map_or(0., |r| r[mode * OUTPUT + j]);
                    1. / (1. + (-(*v + bias)).exp())
                } else {
                    v.max(0.)
                };
            }
        }
        let mut masks: Vec<_> = (0..SLOTS).map(|i| (i, x[i * 50 + 49])).collect();
        let count = masks
            .iter()
            .map(|(_, p)| p)
            .sum::<f32>()
            .round()
            .clamp(1., SLOTS as f32) as usize;
        masks.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut indices: Vec<_> = if self.prefix_masks {
            (0..count).collect()
        } else {
            masks.iter().take(count).map(|(i, _)| *i).collect()
        };
        indices.sort();
        let advance = self
            .advances
            .get(index * self.modes + mode)
            .copied()
            .unwrap_or(1.);
        let (a, b) = ink_layout::font_metrics(label);
        let vertical_scale = if self.canonical_em { a + b } else { 1. };
        let mut strokes = vec![];
        for i in indices {
            let xy = &x[i * 50..i * 50 + 48];
            let mut points = vec![[xy[0] * 1.4 - 0.2, xy[1] * 1.4 - 0.2]];
            for j in 1..POINTS {
                let a = [xy[(j - 1) * 2] * 1.4 - 0.2, xy[(j - 1) * 2 + 1] * 1.4 - 0.2];
                let b = [xy[j * 2] * 1.4 - 0.2, xy[j * 2 + 1] * 1.4 - 0.2];
                if self.prefix_masks {
                    let previous = if j >= 2 {
                        [xy[(j - 2) * 2] * 1.4 - 0.2, xy[(j - 2) * 2 + 1] * 1.4 - 0.2]
                    } else {
                        a
                    };
                    let next = if j + 1 < POINTS {
                        [xy[(j + 1) * 2] * 1.4 - 0.2, xy[(j + 1) * 2 + 1] * 1.4 - 0.2]
                    } else {
                        b
                    };
                    points.extend([
                        [
                            a[0] + (b[0] - previous[0]) / 6.,
                            a[1] + (b[1] - previous[1]) / 6.,
                        ],
                        [b[0] - (next[0] - a[0]) / 6., b[1] - (next[1] - a[1]) / 6.],
                        b,
                    ]);
                } else {
                    points.extend([
                        [a[0] + (b[0] - a[0]) / 3., a[1] + (b[1] - a[1]) / 3.],
                        [
                            a[0] + (b[0] - a[0]) * 2. / 3.,
                            a[1] + (b[1] - a[1]) * 2. / 3.,
                        ],
                        b,
                    ]);
                }
            }
            strokes.push(Stroke {
                points: points
                    .into_iter()
                    .map(|p| [p[0] * advance, p[1] * vertical_scale])
                    .collect(),
                width: (x[i * 50 + 48] * 0.08).max(0.005),
                source_index: i,
                uuid: format!("neural-compositional-{i}"),
            });
        }
        Ok(strokes)
    }
}

#[cfg(test)]
mod contracts {
    use super::*;
    #[test]
    fn x_height_does_not_shrink_canonical_ink_width() {
        let mut bias = vec![0.; OUTPUT];
        for i in 0..SLOTS {
            bias[i * 50 + 49] = if i == 0 { 16. } else { -16. };
        }
        for j in 0..POINTS {
            for (a, v) in [
                j as f32 / (POINTS - 1) as f32 * 0.5,
                j as f32 / (POINTS - 1) as f32,
            ]
            .into_iter()
            .enumerate()
            {
                let u = (v + 0.2) / 1.4;
                bias[j * 2 + a] = (u / (1. - u)).ln();
            }
        }
        let decoder = GlyphDecoder {
            anchor_head: None,
            anchor_conditions: vec![],
            modes: 1,
            preferred_modes: vec![],
            canonical_em: true,
            prefix_masks: true,
            residuals: vec![],
            advances: vec![1., 1.],
            vocabulary: vec!["A".into(), "a".into()],
            supported: vec!["A".into(), "a".into()],
            embedding: vec![vec![0.; 32]; 2],
            layers: vec![
                Linear {
                    input: 48,
                    bias: vec![0.],
                    weight: vec![0.; 48],
                },
                Linear {
                    input: 1,
                    bias: vec![0.],
                    weight: vec![0.],
                },
                Linear {
                    input: 1,
                    bias,
                    weight: vec![0.; OUTPUT],
                },
            ],
        };
        decoder.validate().unwrap();
        let drawing = ink_layout::generate("Aa", 419, &mut |label, seed| {
            Ok(ink_layout::Glyph {
                run_labels: vec![],
                condition_label: Some(label.into()),
                label: label.into(),
                mode: "fixture".into(),
                model_id: "fixture".into(),
                decoder_sha256: "a".repeat(64),
                latent_seed: seed,
                strokes: decoder.generate(label, seed, 0., 419)?,
                transform: [1., 1., 0., 0.],
            })
        })
        .unwrap();
        let extents: Vec<_> = drawing
            .glyphs
            .iter()
            .map(|g| {
                let mut b = [
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ];
                for p in g.strokes.iter().flat_map(|s| &s.points) {
                    let x = p[0] * g.transform[0] + g.transform[2];
                    let y = p[1] * g.transform[1] + g.transform[3];
                    b[0] = b[0].min(x);
                    b[1] = b[1].min(y);
                    b[2] = b[2].max(x);
                    b[3] = b[3].max(y);
                }
                [b[2] - b[0], b[3] - b[1]]
            })
            .collect();
        assert!((extents[0][0] - extents[1][0]).abs() < 1e-5);
        assert!((extents[0][1] - 1.).abs() < 1e-5);
        assert!((extents[1][1] - 0.72).abs() < 1e-5);
    }
}

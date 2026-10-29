//! Inference-only conditional VAE decoder. No training runtime or source samples.
pub mod compositional;
pub mod sequence;
use anyhow::{Context, Result, ensure};
use ink_corpus::Stroke;
use ink_layout::{Drawing, Glyph};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const XY: usize = 384;
pub const OUTPUT: usize = 392;
pub const LATENT: usize = 16;
#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: usize,
}
impl Linear {
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            self.input == x.len() && self.weight.len() == self.bias.len() * self.input,
            "invalid neural layer dimensions"
        );
        Ok(self
            .weight
            .chunks_exact(self.input)
            .zip(&self.bias)
            .map(|(w, b)| w.iter().zip(x).fold(*b, |sum, (w, x)| sum + w * x))
            .collect())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Decoder {
    pub schema_version: u32,
    pub kind: String,
    pub model_id: String,
    pub vocabulary: Vec<String>,
    pub layers: Vec<Linear>,
    pub latent_scale: f32,
}
impl Decoder {
    pub fn load(path: &Path) -> Result<Self> {
        let d: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read neural decoder {}", path.display()))?,
        )?;
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 2 && self.kind == "conditional-vae-v2",
            "model is not a neural VAE; train a new model"
        );
        ensure!(
            !self.vocabulary.is_empty() && self.vocabulary.windows(2).all(|w| w[0] < w[1]),
            "invalid vocabulary"
        );
        ensure!(
            self.vocabulary.iter().all(|s| ink_corpus::glyph_label(s)),
            "invalid glyph label"
        );
        ensure!(
            self.latent_scale.is_finite() && self.latent_scale > 0.,
            "invalid latent scale"
        );
        ensure!(self.layers.len() == 3, "expected three decoder layers");
        let mut n = LATENT + self.vocabulary.len();
        for layer in &self.layers {
            ensure!(
                layer.input == n
                    && !layer.bias.is_empty()
                    && layer.weight.len() == n * layer.bias.len(),
                "neural shape mismatch"
            );
            ensure!(
                layer
                    .weight
                    .iter()
                    .chain(&layer.bias)
                    .all(|x| x.is_finite()),
                "nonfinite weights"
            );
            n = layer.bias.len();
        }
        ensure!(n == OUTPUT, "neural output shape mismatch");
        Ok(())
    }
    pub fn forward(&self, label: &str, latent: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            latent.len() == LATENT && latent.iter().all(|x| x.is_finite()),
            "invalid latent vector"
        );
        let index=self.vocabulary.iter().position(|s|s==label).with_context(||format!("missing trained glyph {label:?}; available: {}. Add labeled atomic ink, then train a new model",self.vocabulary.join(" ")))?;
        let mut x = latent.to_vec();
        x.resize(LATENT + self.vocabulary.len(), 0.);
        x[LATENT + index] = 1.;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.apply(&x)?;
            for v in &mut x {
                *v = if i == 2 {
                    1. / (1. + (-*v).exp())
                } else {
                    v.max(0.)
                };
            }
        }
        ensure!(x.iter().all(|v| v.is_finite()), "nonfinite neural output");
        Ok(x)
    }
    pub fn draw(&self, input: &str, seed: u64, checksum: &str) -> Result<Drawing> {
        // Preflight all labels before any output is written.
        let mut labels = vec![];
        ink_layout::labels(&ink_layout::parse(input)?, &mut labels);
        let missing: std::collections::BTreeSet<_> = labels
            .iter()
            .map(|l| ink_layout::neural_condition(l))
            .filter(|l| {
                !ink_layout::procedural_symbol(l)
                    && !self.vocabulary.iter().any(|s| s.as_str() == *l)
            })
            .collect();
        ensure!(
            missing.is_empty(),
            "missing trained glyphs: {}. Available: {}",
            missing.into_iter().collect::<Vec<_>>().join(" "),
            self.vocabulary.join(" ")
        );
        ink_layout::generate(input, seed, &mut |label, seed| {
            let z = normal(seed, LATENT)
                .into_iter()
                .map(|x| x * self.latent_scale)
                .collect::<Vec<_>>();
            let y = self.forward(label, &z)?;
            Ok(Glyph {
                run_labels: vec![],
                condition_label: None,
                label: label.into(),
                mode: "neural".into(),
                model_id: self.model_id.clone(),
                decoder_sha256: checksum.into(),
                latent_seed: seed,
                strokes: strokes(&y)?,
                transform: [1., 1., 0., 0.],
            })
        })
    }
}
pub fn strokes(y: &[f32]) -> Result<Vec<Stroke>> {
    ensure!(
        y.len() == OUTPUT && y.iter().all(|v| v.is_finite()),
        "invalid neural ink"
    );
    let count = y[XY..].iter().take_while(|p| **p >= 0.5).count().max(1);
    let mut strokes = vec![];
    for i in 0..count {
        let xy = &y[i * 48..(i + 1) * 48];
        let mut points = vec![[xy[0], xy[1]]];
        // Straight cubic segments interpolate learned ordered points exactly.
        for j in 1..24 {
            let a = [xy[(j - 1) * 2], xy[(j - 1) * 2 + 1]];
            let b = [xy[j * 2], xy[j * 2 + 1]];
            points.push([a[0] + (b[0] - a[0]) / 3., a[1] + (b[1] - a[1]) / 3.]);
            points.push([
                a[0] + 2. * (b[0] - a[0]) / 3.,
                a[1] + 2. * (b[1] - a[1]) / 3.,
            ]);
            points.push(b);
        }
        strokes.push(Stroke {
            points,
            source_index: i,
            uuid: format!("neural-stroke-{i}"),
            width: 0.025,
        });
    }
    Ok(strokes)
}
pub fn normal(mut state: u64, count: usize) -> Vec<f32> {
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut n = state;
        n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
        n ^= n >> 31;
        ((n >> 40) as f32 + 0.5) / 16777216.
    };
    let mut out = vec![];
    while out.len() < count {
        let r = (-2. * uniform().ln()).sqrt();
        let t = std::f32::consts::TAU * uniform();
        out.push(r * t.cos());
        if out.len() < count {
            out.push(r * t.sin());
        }
    }
    out
}
/// Dispatches immutable legacy decoders and variable-length sequence decoders.
pub enum Runtime {
    Legacy(Decoder),
    Sequence(Box<sequence::SequenceDecoder>),
}
struct SequenceProvider<'a> {
    decoder: &'a sequence::SequenceDecoder,
    checksum: &'a str,
    seed: u64,
    variation: f32,
}
impl ink_layout::InkProvider for SequenceProvider<'_> {
    fn glyph(&mut self, label: &str, seed: u64) -> Result<Glyph> {
        Ok(Glyph {
            run_labels: vec![],
            condition_label: Some(label.into()),
            label: label.into(),
            mode: "neural".into(),
            model_id: self.decoder.model_id.clone(),
            decoder_sha256: self.checksum.into(),
            latent_seed: seed,
            strokes: self
                .decoder
                .generate(label, seed, self.variation, self.seed)?,
            transform: [1., 1., 0., 0.],
        })
    }
    fn run(&mut self, labels: &[String], seed: u64) -> Result<Option<Glyph>> {
        if !self.decoder.whole_run {
            return Ok(None);
        }
        Ok(Some(Glyph {
            run_labels: labels.to_vec(),
            condition_label: None,
            label: labels
                .iter()
                .map(|s| if s == "<space>" { " " } else { s })
                .collect(),
            mode: "neural-run".into(),
            model_id: self.decoder.model_id.clone(),
            decoder_sha256: self.checksum.into(),
            latent_seed: seed,
            strokes: self
                .decoder
                .generate_tokens(labels, seed, self.variation, self.seed)?,
            transform: [1., 1., 0., 0.],
        }))
    }
}
impl Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        match v["kind"].as_str() {
            Some("conditional-vae-v2") => Ok(Self::Legacy(Decoder::load(path)?)),
            Some("sequence-cvae-v3") => Ok(Self::Sequence(Box::new(
                sequence::SequenceDecoder::load(path)?,
            ))),
            _ => anyhow::bail!("unsupported neural decoder"),
        }
    }
    pub fn vocabulary(&self) -> &[String] {
        match self {
            Self::Legacy(d) => &d.vocabulary,
            Self::Sequence(d) => &d.vocabulary,
        }
    }
    pub fn model_id(&self) -> &str {
        match self {
            Self::Legacy(d) => &d.model_id,
            Self::Sequence(d) => &d.model_id,
        }
    }
    pub fn draw(&self, input: &str, seed: u64, checksum: &str, variation: f32) -> Result<Drawing> {
        ensure!(
            variation.is_finite() && (0.0..=2.0).contains(&variation),
            "variation must be 0..2"
        );
        let drawing = match self {
            Self::Legacy(d) => {
                let mut d = d.clone();
                d.latent_scale *= variation;
                d.draw(input, seed, checksum)?
            }
            Self::Sequence(d) => {
                let ast = ink_layout::parse(input)?;
                let mut labels = vec![];
                ink_layout::labels(&ast, &mut labels);
                for label in &labels {
                    if !ink_layout::procedural_symbol(label) {
                        d.condition(ink_layout::neural_condition(label))?;
                    }
                }
                let mut drawing = if d.whole_run {
                    let conditions = ink_layout::tokenize(input)?;
                    let glyph = Glyph {
                        run_labels: labels,
                        condition_label: None,
                        label: input.into(),
                        mode: if d.style_domain == "public" {
                            "neural-run-public"
                        } else {
                            "neural-run"
                        }
                        .into(),
                        model_id: d.model_id.clone(),
                        decoder_sha256: checksum.into(),
                        latent_seed: seed,
                        strokes: d.generate_tokens(&conditions, seed, variation, seed)?,
                        transform: [1., 1., 0., 0.],
                    };
                    ink_layout::generate_sequence(input, seed, glyph)?
                } else {
                    let mut provider = SequenceProvider {
                        decoder: d,
                        checksum,
                        seed,
                        variation,
                    };
                    ink_layout::generate_with_provider(input, seed, &mut provider)?
                };
                drawing.schema_version = 3;
                drawing
            }
        };
        Ok(drawing)
    }
}
pub fn random_seed() -> Result<u64> {
    use std::io::Read;
    let mut bytes = [0_u8; 8];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_neural_points_survive_cubic_conversion() {
        let mut y = vec![0.25; OUTPUT];
        y[XY] = 1.;
        y[XY + 1] = 1.;
        let s = strokes(&y).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].points.len(), 70);
        assert_eq!(s[0].points[69], [0.25, 0.25]);
        assert_eq!(normal(419, 16), normal(419, 16));
        assert_ne!(normal(419, 16), normal(420, 16));
    }
}

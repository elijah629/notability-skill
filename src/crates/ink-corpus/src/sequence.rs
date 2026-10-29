//! Lossless, reversible sample-level cubic events. No character segmentation.
use crate::{Sample, Stroke, bounds};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
pub const FEATURES: usize = 10;
pub const MOVE: usize = 7;
pub const CURVE: usize = 8;
pub const END: usize = 9;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sequence {
    pub events: Vec<[f32; FEATURES]>,
    pub origin: [f32; 2],
    pub scale: f32,
}
/// Balance native pen strokes, not inferred characters. Short T crossbars and
/// dots must not disappear behind densely sampled long strokes. EOS is one group.
pub fn training_weights(events: &[[f32; FEATURES]]) -> Result<Vec<f32>> {
    ensure!(
        events.len() >= 2 && events[0][MOVE] > 0.5 && events.last().is_some_and(|e| e[END] > 0.5),
        "training weights require a complete native sequence"
    );
    let mut boundaries: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| e[MOVE] > 0.5 || e[END] > 0.5)
        .map(|(i, _)| i)
        .collect();
    boundaries.push(events.len());
    let mass = events.len() as f32 / (boundaries.len() - 1) as f32;
    let mut weights = vec![0.; events.len()];
    for group in boundaries.windows(2) {
        weights[group[0]..group[1]].fill(0.5 + 0.5 * mass / (group[1] - group[0]) as f32);
    }
    Ok(weights)
}
/// Balance observed event classes without inferring character boundaries.
pub fn event_weights(events: &[[f32; FEATURES]]) -> Result<Vec<f32>> {
    let original = training_weights(events)?;
    let counts: Vec<_> = (MOVE..=END)
        .map(|k| events.iter().filter(|e| e[k] > 0.5).count())
        .collect();
    let classes = counts.iter().filter(|&&n| n > 0).count() as f32;
    Ok(events
        .iter()
        .zip(original)
        .map(|(e, w)| {
            let kind = (MOVE..=END).find(|&k| e[k] > 0.5).expect("native event");
            0.5 * w + 0.5 * events.len() as f32 / (classes * counts[kind - MOVE] as f32)
        })
        .collect())
}
pub fn canonical_feedback(mut event: [f32; FEATURES]) -> [f32; FEATURES] {
    if event[END] > 0.5 {
        event[..7].fill(0.);
    } else {
        if event[MOVE] > 0.5 {
            event[2..6].fill(0.);
        }
        event[6] = event[6].abs().clamp(1e-5, 1.);
    }
    event
}
pub fn encode(sample: &Sample) -> Result<Sequence> {
    encode_scaled(sample, true)
}
/// Reversible em units: repeated text does not shrink individual pen motions.
/// `written_tokens` is a supplied text count, never a character/stroke alignment.
pub fn encode_em(sample: &Sample, written_tokens: usize) -> Result<Sequence> {
    ensure!(written_tokens > 0, "empty written token count");
    let mut encoded = encode(sample)?;
    let b = bounds(sample);
    let scale = (b[3] - b[1])
        .max((b[2] - b[0]) / written_tokens as f32)
        .max(sample.strokes.iter().map(|s| s.width).fold(0_f32, f32::max))
        .max(1e-6);
    let gain = encoded.scale / scale;
    for event in &mut encoded.events {
        for value in &mut event[..7] {
            *value *= gain;
        }
    }
    encoded.scale = scale;
    Ok(encoded)
}
/// Historical recipe compatibility for evaluating v1–v4 checkpoints only.
/// New training uses `encode`; this path retains the original metric definition.
pub fn encode_legacy_control_bounds(sample: &Sample) -> Result<Sequence> {
    encode_scaled(sample, false)
}
fn encode_scaled(sample: &Sample, include_pen_width: bool) -> Result<Sequence> {
    let b = bounds(sample);
    // A native tap can have zero coordinate extent while still carrying a real
    // pen diameter. Include that diameter in the isotropic scale rather than
    // dividing its width by epsilon and creating a huge likelihood target.
    let pen_width = if include_pen_width {
        sample.strokes.iter().map(|s| s.width).fold(0_f32, f32::max)
    } else {
        0.
    };
    let scale = (b[2] - b[0]).max(b[3] - b[1]).max(pen_width).max(1e-6);
    ensure!(scale.is_finite(), "invalid sequence bounds");
    let origin = [b[0], b[1]];
    let mut events = vec![];
    let mut last = [0., 0.];
    for stroke in &sample.strokes {
        ensure!(
            !stroke.points.is_empty() && (stroke.points.len() - 1) % 3 == 0,
            "invalid source cubic"
        );
        let p = [
            (stroke.points[0][0] - origin[0]) / scale,
            (stroke.points[0][1] - origin[1]) / scale,
        ];
        let mut event = [0.; FEATURES];
        event[0] = p[0] - last[0];
        event[1] = p[1] - last[1];
        event[6] = stroke.width / scale;
        event[MOVE] = 1.;
        events.push(event);
        last = p;
        for controls in stroke.points[1..].as_chunks::<3>().0 {
            let mut event = [0.; FEATURES];
            for (i, p) in controls.iter().enumerate() {
                for a in 0..2 {
                    event[i * 2 + a] = (p[a] - origin[a]) / scale - last[a];
                }
            }
            event[6] = stroke.width / scale;
            event[CURVE] = 1.;
            last = [last[0] + event[4], last[1] + event[5]];
            events.push(event);
        }
    }
    let mut end = [0.; FEATURES];
    end[END] = 1.;
    events.push(end);
    ensure!(
        events.iter().flatten().all(|x| x.is_finite()),
        "nonfinite sequence"
    );
    Ok(Sequence {
        events,
        origin,
        scale,
    })
}
pub fn decode(events: &[[f32; FEATURES]], origin: [f32; 2], scale: f32) -> Result<Vec<Stroke>> {
    ensure!(
        scale.is_finite() && scale > 0. && origin.iter().all(|x| x.is_finite()),
        "invalid inverse transform"
    );
    let mut strokes: Vec<Stroke> = vec![];
    let mut last = [0., 0.];
    let mut ended = false;
    for e in events {
        ensure!(e.iter().all(|x| x.is_finite()), "nonfinite event");
        if e[END] > 0.5 {
            ended = true;
            break;
        }
        if e[MOVE] > 0.5 {
            last = [last[0] + e[0], last[1] + e[1]];
            let i = strokes.len();
            strokes.push(Stroke {
                points: vec![[last[0] * scale + origin[0], last[1] * scale + origin[1]]],
                source_index: i,
                uuid: format!("neural-stroke-{i}"),
                width: (e[6] * scale).max(0.0001),
            });
        } else {
            ensure!(e[CURVE] > 0.5, "invalid event kind");
            let stroke = strokes
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("curve before move"))?;
            for i in 0..3 {
                stroke.points.push([
                    (last[0] + e[i * 2]) * scale + origin[0],
                    (last[1] + e[i * 2 + 1]) * scale + origin[1],
                ]);
            }
            last = [last[0] + e[4], last[1] + e[5]];
        }
    }
    ensure!(
        ended && !strokes.is_empty(),
        "sequence did not terminate with ink"
    );
    Ok(strokes)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn em_units_preserve_pen_motions_when_text_is_repeated() {
        let one = Sample {
            sample_id: "em-fixture".into(),
            label: "x".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: serde_json::Value::Null,
            strokes: vec![Stroke {
                points: vec![[0., 0.], [0.2, 1.], [0.3, 0.], [0.5, 0.5]],
                width: 0.02,
                source_index: 0,
                uuid: "fixture".into(),
            }],
        };
        let mut repeated = one.clone();
        repeated.label = "xx".into();
        let mut second = one.strokes[0].clone();
        for p in &mut second.points {
            p[0] += 0.75;
        }
        repeated.strokes.push(second);
        assert_ne!(
            encode(&one).unwrap().events[1],
            encode(&repeated).unwrap().events[1],
            "legacy whole-width units hide the regression"
        );
        let a = encode_em(&one, 1).unwrap();
        let b = encode_em(&repeated, 2).unwrap();
        for (x, y) in a.events[1].iter().zip(b.events[1]) {
            assert!((x - y).abs() < 1e-6);
        }
        let restored = decode(&b.events, b.origin, b.scale).unwrap();
        for (original, restored) in repeated.strokes.iter().zip(restored) {
            for (p, q) in original.points.iter().zip(restored.points) {
                for axis in 0..2 {
                    assert!((p[axis] - q[axis]).abs() < 1e-6);
                }
            }
            assert!((original.width - restored.width).abs() < 1e-6);
        }
        assert!(encode_em(&one, 0).is_err());
    }
    #[test]
    fn rollout_feedback_preserves_native_event_semantics() {
        let mut e = [0.3; FEATURES];
        e[MOVE] = 1.;
        e[CURVE] = 0.;
        e[END] = 0.;
        e[6] = -0.02;
        let move_event = canonical_feedback(e);
        assert_eq!(&move_event[..6], &[0.3, 0.3, 0., 0., 0., 0.]);
        assert_eq!(move_event[6], 0.02);
        e[MOVE] = 0.;
        e[CURVE] = 1.;
        assert_eq!(&canonical_feedback(e)[..6], &e[..6]);
        e[CURVE] = 0.;
        e[END] = 1.;
        assert_eq!(&canonical_feedback(e)[..7], &[0.; 7]);
    }
    #[test]
    fn event_balancing_retains_finite_mass_for_sparse_lifts_and_eos() {
        let mut events = vec![[0.; FEATURES]; 102];
        events[0][MOVE] = 1.;
        for event in &mut events[1..101] {
            event[CURVE] = 1.;
        }
        events[101][END] = 1.;
        let weights = event_weights(&events).unwrap();
        assert!(weights.iter().all(|w| w.is_finite() && *w > 0.));
        assert!((weights.iter().sum::<f32>() - 102.).abs() < 1e-3);
        assert!(weights[0] > weights[1] * 10.);
        assert!(weights[101] > weights[1] * 10.);
    }
    #[test]
    fn point_stroke_scale_keeps_pen_width_bounded_and_round_trips() {
        let sample = Sample {
            sample_id: "native-dot".into(),
            label: ".".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: serde_json::Value::Null,
            strokes: vec![Stroke {
                points: vec![[10., 20.]],
                width: 0.02,
                source_index: 0,
                uuid: "original-dot".into(),
            }],
        };
        let encoded = encode(&sample).unwrap();
        assert_eq!(
            encode_legacy_control_bounds(&sample).unwrap().events[0][6],
            20_000.
        );
        assert!(
            encoded.events[0][6] <= 1.,
            "normalized pen width exploded: {}",
            encoded.events[0][6]
        );
        let decoded = decode(&encoded.events, encoded.origin, encoded.scale).unwrap();
        assert_eq!(decoded[0].points, sample.strokes[0].points);
        assert!((decoded[0].width - sample.strokes[0].width).abs() < 1e-7);
    }
    #[test]
    fn variable_cubics_round_trip_without_truncation() {
        let s = Sample {
            sample_id: "fixture".into(),
            label: "whole sentence".into(),
            source_sha256: "a".repeat(64),
            split: "train".into(),
            metadata: serde_json::Value::Null,
            strokes: (0..94)
                .map(|i| Stroke {
                    points: vec![
                        [i as f32, 2.],
                        [i as f32 + 0.1, 2.5],
                        [i as f32 + 0.3, 2.7],
                        [i as f32 + 0.5, 3.],
                    ],
                    width: 0.1,
                    source_index: i,
                    uuid: i.to_string(),
                })
                .collect(),
        };
        let q = encode(&s).unwrap();
        let r = decode(&q.events, q.origin, q.scale).unwrap();
        assert_eq!(r.len(), 94);
        for (a, b) in s.strokes.iter().zip(r) {
            for (p, q) in a.points.iter().zip(b.points) {
                assert!((p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4);
            }
        }
        assert!(decode(&q.events[..q.events.len() - 1], q.origin, q.scale).is_err());
    }
    #[test]
    fn short_crossbars_retain_loss_mass_without_character_segmentation() {
        let mut events = vec![[0.; FEATURES]; 104];
        events[0][MOVE] = 1.;
        for e in &mut events[1..101] {
            e[CURVE] = 1.;
        }
        events[101][MOVE] = 1.;
        events[102][CURVE] = 1.;
        events[103][END] = 1.;
        let weights = training_weights(&events).unwrap();
        assert!(weights[101] > weights[0] * 10.);
        assert!((weights.iter().sum::<f32>() - events.len() as f32).abs() < 1e-3);
        let mass = |a: usize, b: usize| weights[a..b].iter().sum::<f32>() - (b - a) as f32 * 0.5;
        assert!((mass(0, 101) - mass(101, 103)).abs() < 1e-3);
        assert!(training_weights(&events[..103]).is_err());
    }
}

pub mod additive;
mod parquet;
pub mod sequence;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Cursor, Read, Write},
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stroke {
    pub points: Vec<[f32; 2]>,
    pub source_index: usize,
    pub uuid: String,
    pub width: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub sample_id: String,
    pub label: String,
    pub source_sha256: String,
    pub split: String,
    pub strokes: Vec<Stroke>,
    pub metadata: Value,
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file_hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn save(path: &Path, v: &impl Serialize) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
pub fn load(path: &Path) -> Result<Vec<Sample>> {
    BufReader::new(fs::File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(&l?).with_context(|| format!("{} line {}", path.display(), i + 1))
        })
        .collect()
}
pub fn glyph_label(label: &str) -> bool {
    (label.chars().count() == 1
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "+-=()[]<>.,:;!|/".contains(c)))
        || [
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\delta",
            "\\epsilon",
            "\\theta",
            "\\lambda",
            "\\mu",
            "\\pi",
            "\\rho",
            "\\sigma",
            "\\tau",
            "\\phi",
            "\\omega",
            "\\Delta",
            "\\Sigma",
            "\\Omega",
            "\\infty",
            "\\times",
            "\\cdot",
            "\\pm",
            "\\leq",
            "\\geq",
            "\\neq",
            "\\in",
            "\\partial",
            "\\nabla",
            "\\sum",
            "\\int",
            "\\div",
            "\\approx",
            "\\cup",
            "\\cap",
            "\\eta",
            "\\upsilon",
            "\\Lambda",
            "\\Phi",
            "\\Theta",
            "\\Pi",
            "\\rightarrow",
            "\\leftarrow",
            "\\downarrow",
            "\\parallel",
            "\\circ",
            "\\angle",
            "\\sim",
            "\\propto",
            "\\backslash",
        ]
        .contains(&label)
}
fn member(z: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<Vec<u8>> {
    let mut f = z
        .by_name(name)
        .with_context(|| format!("dataset missing {name}"))?;
    ensure!(f.size() <= 256_000_000, "dataset member too large");
    let mut b = Vec::new();
    f.read_to_end(&mut b)?;
    Ok(b)
}
fn npy<'a>(b: &'a [u8], dtype: &str) -> Result<(&'a [u8], Vec<usize>)> {
    ensure!(b.len() >= 10 && &b[..6] == b"\x93NUMPY", "invalid NPY");
    let (start, len) = match b[6] {
        1 => (10, u16::from_le_bytes(b[8..10].try_into()?) as usize),
        2 | 3 => {
            ensure!(b.len() >= 12, "truncated NPY");
            (12, u32::from_le_bytes(b[8..12].try_into()?) as usize)
        }
        _ => bail!("unsupported NPY version"),
    };
    ensure!(start + len <= b.len(), "truncated NPY header");
    let h = std::str::from_utf8(&b[start..start + len])?;
    ensure!(
        h.contains(&format!("'{dtype}'")) && h.contains("'fortran_order': False"),
        "unsupported NPY dtype/order: {h}"
    );
    let shape = h
        .split("'shape':")
        .nth(1)
        .context("NPY shape missing")?
        .split('(')
        .nth(1)
        .context("NPY shape")?
        .split(')')
        .next()
        .context("NPY shape")?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let count = dims.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b).context("NPY shape overflow")
    })?;
    let size = match dtype {
        "<f4" => 4,
        "|V16" | "|S16" => 16,
        "|u1" => 1,
        _ => 8,
    };
    ensure!(
        count.checked_mul(size) == Some(b.len() - start - len),
        "NPY size mismatch"
    );
    Ok((&b[start + len..], dims))
}
fn offsets(z: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<Vec<usize>> {
    let b = member(z, name)?;
    let (data, dims) = npy(&b, "<u8")?;
    ensure!(dims.len() == 1, "offsets rank");
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|c| Ok(usize::try_from(u64::from_le_bytes(*c))?))
        .collect()
}
pub fn import(archive: &Path, out: &Path) -> Result<Value> {
    ensure!(!out.exists(), "import requires fresh output directory");
    let bytes = fs::read(archive)?;
    let dataset_hash = hash(&bytes);
    let mut z = zip::ZipArchive::new(Cursor::new(bytes.as_slice()))?;
    let manifest = String::from_utf8(member(&mut z, "MANIFEST.sha256")?)?;
    let mut verified = BTreeSet::new();
    for l in manifest.lines() {
        let (expected, name) = l.split_once("  ").context("invalid SHA256 manifest")?;
        ensure!(
            expected.len() == 64 && verified.insert(name.to_owned()),
            "invalid/duplicate manifest entry"
        );
        ensure!(
            hash(&member(&mut z, name)?) == expected,
            "checksum mismatch: {name}"
        );
    }
    for required in [
        "samples.parquet",
        "vectors.npz",
        "audit/stroke_map.jsonl",
        "audit/sample_lineage.jsonl",
        "statistics.json",
    ] {
        ensure!(
            verified.contains(required),
            "unverified required file {required}"
        );
    }
    let rows = parquet::rows(&member(&mut z, "samples.parquet")?)?;
    let vector_bytes = member(&mut z, "vectors.npz")?;
    let mut vectors = zip::ZipArchive::new(Cursor::new(vector_bytes.as_slice()))?;
    let b = member(&mut vectors, "points.npy")?;
    let (data, dims) = npy(&b, "<f4")?;
    ensure!(
        dims.len() == 2 && dims[1] == 2,
        "points must have shape [P,2]"
    );
    let points: Vec<[f32; 2]> = data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            [
                f32::from_le_bytes(c[..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..].try_into().unwrap()),
            ]
        })
        .collect();
    ensure!(
        points.iter().flatten().all(|x| x.is_finite()),
        "nonfinite source geometry"
    );
    let so = offsets(&mut vectors, "stroke_offsets.npy")?;
    let sa = offsets(&mut vectors, "sample_offsets.npy")?;
    ensure!(
        so.first() == Some(&0)
            && so.last() == Some(&points.len())
            && so.windows(2).all(|w| w[0] < w[1]),
        "invalid stroke offsets"
    );
    ensure!(
        sa.len() == rows.len() + 1
            && sa.first() == Some(&0)
            && sa.last() == Some(&(so.len() - 1))
            && sa.windows(2).all(|w| w[0] < w[1]),
        "invalid sample offsets"
    );
    let maps: Vec<Value> = String::from_utf8(member(&mut z, "audit/stroke_map.jsonl")?)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(maps.len() + 1 == so.len(), "stroke map length mismatch");
    let lineage: Vec<Value> = String::from_utf8(member(&mut z, "audit/sample_lineage.jsonl")?)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let lineage: BTreeMap<String, Value> = lineage
        .into_iter()
        .map(|v| {
            Ok((
                v["sample_id"]
                    .as_str()
                    .context("lineage sample ID")?
                    .to_owned(),
                v,
            ))
        })
        .collect::<Result<_>>()?;
    let mut samples = Vec::new();
    for (i, row) in rows.into_iter().enumerate() {
        let r = serde_json::to_value(row)?;
        let s = |key: &str| -> Result<String> {
            Ok(r[key]
                .as_str()
                .with_context(|| format!("missing sample {key}"))?
                .to_owned())
        };
        let id = s("sample_id")?;
        let source = s("source_sha256")?;
        let label = s("latex")?;
        let split = s("split")?;
        ensure!(
            r["stroke_start"].as_u64() == Some(sa[i] as u64)
                && r["stroke_count"].as_u64() == Some((sa[i + 1] - sa[i]) as u64),
            "sample offset metadata mismatch"
        );
        let lin = lineage.get(&id).context("missing sample lineage")?;
        ensure!(lin["source_sha256"] == source, "lineage source mismatch");
        let indices: Vec<usize> = serde_json::from_str(
            lin["original_stroke_indices"]
                .as_str()
                .context("lineage indices")?,
        )?;
        let mut strokes = Vec::new();
        for j in sa[i]..sa[i + 1] {
            let m = &maps[j];
            ensure!(
                m["packed_stroke_index"].as_u64() == Some(j as u64)
                    && m["sample_id"] == id
                    && m["source_sha256"] == source
                    && m["start_point_index"].as_u64() == Some(so[j] as u64)
                    && m["point_count"].as_u64() == Some((so[j + 1] - so[j]) as u64),
                "stroke lineage mismatch"
            );
            let p = points[so[j]..so[j + 1]].to_vec();
            ensure!(p.len() % 3 == 1, "invalid cubic stroke");
            strokes.push(Stroke {
                points: p,
                source_index: usize::try_from(
                    m["source_stroke_index"]
                        .as_u64()
                        .context("source stroke index")?,
                )?,
                uuid: m["stroke_uuid_hex"]
                    .as_str()
                    .context("stroke UUID")?
                    .to_owned(),
                width: m["pen_width"].as_f64().context("pen width")? as f32,
            });
        }
        ensure!(
            indices == strokes.iter().map(|s| s.source_index).collect::<Vec<_>>(),
            "stroke order differs from annotation"
        );
        samples.push(Sample{sample_id:id,label,source_sha256:source,split,strokes,metadata:json!({"original":r,"annotation":lin,"review_status":"supplied-worker-reviewed; independent second pass absent","authorship":"supplied personal dataset; not independently certified"})});
    }
    let report = audit(&samples)?;
    let statistics: Value = serde_json::from_slice(&member(&mut z, "statistics.json")?)?;
    ensure!(
        statistics["accepted_samples_total"].as_u64() == Some(samples.len() as u64),
        "statistics sample mismatch"
    );
    fs::create_dir_all(out)?;
    let mut f = fs::File::create(out.join("corpus.jsonl"))?;
    for s in &samples {
        serde_json::to_writer(&mut f, s)?;
        writeln!(f)?;
    }
    let mut tables = BTreeMap::new();
    for name in ["sources", "exclusions", "candidate_status", "annotations"] {
        let table = parquet::rows(&member(&mut z, &format!("{name}.parquet"))?)?;
        tables.insert(name, table.len());
        let mut output = fs::File::create(out.join(format!("{name}.jsonl")))?;
        for row in table {
            serde_json::to_writer(&mut output, &row)?;
            writeln!(output)?;
        }
    }
    let report = json!({"dataset_sha256":dataset_hash,"corpus_sha256":file_hash(&out.join("corpus.jsonl"))?,"verified_archive_members":verified.len(),"metadata_table_rows":tables,"audit":report,"source_statistics":statistics,"policy":"User-supplied full dataset; rebuild and independent review waived by user. No inferred glyph segmentation.","limitations":["independent second visual pass absent","original page transforms unresolved"]});
    save(&out.join("manifest.json"), &report)?;
    Ok(report)
}
pub fn audit(samples: &[Sample]) -> Result<Value> {
    let mut ids = BTreeSet::new();
    let mut ownership = BTreeSet::new();
    let mut groups = BTreeMap::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut glyphs = BTreeMap::<String, usize>::new();
    let mut oversized = 0;
    for s in samples {
        ensure!(ids.insert(&s.sample_id), "duplicate sample ID");
        ensure!(!s.label.is_empty() && !s.strokes.is_empty(), "empty sample");
        ensure!(
            ["train", "validation", "test"].contains(&s.split.as_str()),
            "unknown split"
        );
        ensure!(s.source_sha256.len() == 64, "invalid source hash");
        if let Some(prev) = groups.insert(&s.source_sha256, &s.split) {
            ensure!(prev == &s.split, "source crosses split");
        }
        *counts.entry(s.split.clone()).or_default() += 1;
        if glyph_label(&s.label) {
            *glyphs.entry(s.label.clone()).or_default() += 1;
            if s.strokes.len() > 8 {
                oversized += 1;
            }
        }
        for stroke in &s.strokes {
            ensure!(
                ownership.insert((&s.source_sha256, stroke.source_index)),
                "duplicate source stroke"
            );
            ensure!(
                !stroke.points.is_empty()
                    && stroke.points.len() % 3 == 1
                    && stroke.points.iter().flatten().all(|x| x.is_finite())
                    && stroke.width.is_finite()
                    && stroke.width > 0.,
                "invalid curve/width"
            );
        }
    }
    for split in ["train", "validation", "test"] {
        ensure!(
            counts.get(split).copied().unwrap_or(0) > 0,
            "empty {split} partition"
        );
    }
    Ok(
        json!({"samples":samples.len(),"source_groups":groups.len(),"strokes":ownership.len(),"splits":counts,"atomic_glyph_samples":glyphs.values().sum::<usize>(),"glyph_coverage":glyphs,"oversized_glyphs_excluded_from_tensors":oversized,"source_split_leakage":0,"duplicate_stroke_count":0}),
    )
}
/// Derived 8 x 24 x XY tensor; original cubic controls remain untouched.
pub fn tensor(s: &Sample) -> Result<(Vec<f32>, [f32; 8])> {
    ensure!(
        glyph_label(&s.label),
        "whole phrases/formulas cannot train glyph model"
    );
    ensure!(
        s.strokes.len() <= 8,
        "oversized glyph exceeds eight strokes; preserved in corpus, excluded from neural tensors"
    );
    let bounds = bounds(s);
    let scale = (bounds[2] - bounds[0]).max(bounds[3] - bounds[1]).max(1.);
    let mut xy = vec![0.; 8 * 24 * 2];
    let mut mask = [0.; 8];
    for (i, stroke) in s.strokes.iter().enumerate() {
        mask[i] = 1.;
        let segments = (stroke.points.len() - 1) / 3;
        for j in 0..24 {
            let p = if segments == 0 {
                stroke.points[0]
            } else {
                let u = j as f32 / 23. * segments as f32;
                let k = (u.floor() as usize).min(segments - 1);
                let t = (u - k as f32).min(1.);
                let p = &stroke.points[k * 3..k * 3 + 4];
                let q = 1. - t;
                [
                    q * q * q * p[0][0]
                        + 3. * q * q * t * p[1][0]
                        + 3. * q * t * t * p[2][0]
                        + t * t * t * p[3][0],
                    q * q * q * p[0][1]
                        + 3. * q * q * t * p[1][1]
                        + 3. * q * t * t * p[2][1]
                        + t * t * t * p[3][1],
                ]
            };
            for axis in 0..2 {
                xy[(i * 24 + j) * 2 + axis] = (p[axis] - bounds[axis]) / scale;
            }
        }
    }
    Ok((xy, mask))
}
pub fn bounds(s: &Sample) -> [f32; 4] {
    let mut b = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for p in s.strokes.iter().flat_map(|s| &s.points) {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}
/// Confidence warnings retain supplied labels and vectors; they are not corrected annotations.
pub fn label_geometry_warnings(samples: &[Sample]) -> Vec<Value> {
    samples.iter().filter_map(|s|{let label=s.label.trim();let label=["\\text{","\\mathrm{","\\mathbb{"].iter().find_map(|prefix|label.strip_prefix(prefix).and_then(|s|s.strip_suffix('}'))).unwrap_or(label);if label!="0"||s.strokes.len()!=1{return None;}let points=&s.strokes[0].points;let(a,z)=(points.first()?,points.last()?);let b=bounds(s);let gap=((a[0]-z[0]).powi(2)+(a[1]-z[1]).powi(2)).sqrt();if gap>(b[3]-b[1]).max(0.01)*0.3{Some(json!({"sample_id":s.sample_id,"split":s.split,"label":s.label,"warning":"open path inconsistent with closed-loop zero; supplied label remains unchanged","training":"whole-row supervision retained; strong ordered anchor disabled","transcription_certified":false}))}else{None}}).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn npy_rejects_truncation() {
        assert!(npy(b"bad", "<f4").is_err());
    }
    #[test]
    fn never_guess_segmentation() {
        assert!(glyph_label("x"));
        assert!(glyph_label("\\alpha"));
        assert!(!glyph_label("xy"));
        assert!(!glyph_label("\\text{word}"));
        assert!(!glyph_label("x^2"));
    }
}

#[cfg(test)]
mod contracts {
    use super::*;
    fn sample(id: &str, source: char, split: &str) -> Sample {
        Sample {
            sample_id: id.into(),
            label: "x".into(),
            source_sha256: source.to_string().repeat(64),
            split: split.into(),
            strokes: vec![Stroke {
                points: vec![[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
                source_index: 0,
                uuid: id.into(),
                width: 0.02,
            }],
            metadata: json!({"fixture":true}),
        }
    }
    #[test]
    fn source_split_and_stroke_identity_are_enforced() {
        let mut data = vec![
            sample("a", 'a', "train"),
            sample("b", 'b', "validation"),
            sample("c", 'c', "test"),
        ];
        assert!(audit(&data).is_ok());
        let mut duplicate = data[0].clone();
        duplicate.sample_id = "d".into();
        data.push(duplicate);
        assert!(audit(&data).is_err());
        data.pop();
        data[2].source_sha256 = data[0].source_sha256.clone();
        data[2].strokes[0].source_index = 1;
        assert!(audit(&data).is_err());
    }
    #[test]
    fn cubic_controls_survive_tensor_derivation() {
        let s = sample("a", 'a', "train");
        let before = serde_json::to_vec(&s).unwrap();
        let (xy, mask) = tensor(&s).unwrap();
        assert_eq!(&xy[..2], &[0., 0.]);
        assert_eq!(&xy[46..48], &[1., 0.]);
        assert!(xy[22] > 0.4 && xy[22] < 0.6);
        assert!(xy[23] > 0.7);
        assert_eq!(mask, [1., 0., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(before, serde_json::to_vec(&s).unwrap());
        let mut oversized = s;
        oversized.strokes = vec![oversized.strokes[0].clone(); 9];
        assert!(tensor(&oversized).is_err());
    }
}

use super::*;

/// Merge supplied labels and original native curves into an existing corpus.
pub fn merge(input: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    ensure!(!out.exists(), "merge requires fresh output directory");
    let bytes = fs::read(input)?;
    let mut z = zip::ZipArchive::new(Cursor::new(bytes.as_slice()))?;
    let inventory = String::from_utf8(member(&mut z, "PATCH_MANIFEST.sha256")?)?;
    let mut verified = BTreeSet::new();
    for line in inventory.lines() {
        let (expected, name) = line.split_once("  ").context("invalid manifest")?;
        ensure!(
            verified.insert(name.to_owned()),
            "duplicate manifest member"
        );
        ensure!(
            hash(&member(&mut z, name)?) == expected,
            "checksum mismatch: {name}"
        );
    }
    for name in ["PATCH.json", "transcriptions.jsonl", "source_cache.npz"] {
        ensure!(verified.contains(name), "unverified {name}");
    }
    let meta: Value = serde_json::from_slice(&member(&mut z, "PATCH.json")?)?;
    let cache = member(&mut z, "source_cache.npz")?;
    let labels = member(&mut z, "transcriptions.jsonl")?;
    ensure!(
        hash(&cache) == meta["source_cache_sha256"],
        "cache hash mismatch"
    );
    ensure!(
        hash(&labels) == meta["transcriptions_sha256"],
        "labels hash mismatch"
    );
    let source = meta["source_sha256"].as_str().context("source hash")?;
    ensure!(
        source.len() == 64 && source.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid source hash"
    );
    let mut samples = load(&corpus.join("corpus.jsonl"))?;
    let original_count = samples.len();
    let mut manifest: Value = serde_json::from_slice(&fs::read(corpus.join("manifest.json"))?)?;
    ensure!(
        manifest["dataset_sha256"] == meta["baseline_zip_sha256"],
        "baseline mismatch"
    );
    ensure!(
        file_hash(&corpus.join("corpus.jsonl"))? == manifest["corpus_sha256"],
        "corpus hash mismatch"
    );
    let sources = fs::read_to_string(corpus.join("sources.jsonl"))?;
    for line in sources.lines() {
        let row: Value = serde_json::from_str(line)?;
        ensure!(row["sha256"] != source, "source already imported");
    }
    ensure!(
        !samples.iter().any(|s| s.source_sha256 == source),
        "duplicate source"
    );
    let mut native = zip::ZipArchive::new(Cursor::new(cache.as_slice()))?;
    let p = member(&mut native, "points.npy")?;
    let (data, shape) = npy(&p, "<f4")?;
    ensure!(shape.len() == 2 && shape[1] == 2, "invalid point shape");
    let points: Vec<[f32; 2]> = data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|p| {
            [
                f32::from_le_bytes(p[..4].try_into().unwrap()),
                f32::from_le_bytes(p[4..].try_into().unwrap()),
            ]
        })
        .collect();
    ensure!(
        points.iter().flatten().all(|p| p.is_finite()),
        "nonfinite points"
    );
    let offsets = offsets(&mut native, "stroke_offsets.npy")?;
    let n = offsets.len().checked_sub(1).context("empty offsets")?;
    ensure!(
        offsets.first() == Some(&0)
            && offsets.last() == Some(&points.len())
            && offsets.windows(2).all(|w| w[0] < w[1]),
        "invalid native offsets"
    );
    ensure!(
        meta["source_strokes"].as_u64() == Some(n as u64)
            && meta["source_points"].as_u64() == Some(points.len() as u64),
        "source counts mismatch"
    );
    let widths = member(&mut native, "widths.npy")?;
    let (widths, shape) = npy(&widths, "<f4")?;
    ensure!(shape == [n], "width count mismatch");
    let uuids = member(&mut native, "stroke_uuids.npy")?;
    // Native UUID arrays store fixed-width opaque bytes, never text conversions.
    let (uuid_data, shape) = npy(&uuids, "|V16")
        .or_else(|_| npy(&uuids, "|S16"))
        .or_else(|_| npy(&uuids, "|u1"))?;
    ensure!(
        (shape == [n] || shape == [n, 16]) && uuid_data.len() == n * 16,
        "UUID count mismatch"
    );
    ensure!(
        uuid_data
            .as_chunks::<16>()
            .0
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            == n,
        "duplicate native UUID"
    );
    ensure!(
        widths.as_chunks::<4>().0.iter().all(|b| {
            let w = f32::from_le_bytes(*b);
            w.is_finite() && w > 0.
        }),
        "invalid native width"
    );
    let rows: Vec<Value> = String::from_utf8(labels)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let mut ownership = BTreeSet::new();
    let mut annotations = Vec::new();
    let mut decisions = Vec::new();
    let mut exclusions = Vec::new();
    let mut accepted_points = 0usize;
    let mut accepted_strokes = 0usize;
    for row in &rows {
        ensure!(
            row["source_sha256"] == source && row["split"] == "train",
            "source/split mismatch"
        );
        let status = row["status"].as_str().context("status")?;
        ensure!(
            ["accepted", "unresolved"].contains(&status),
            "unsupported status"
        );
        let indices: Vec<usize> = serde_json::from_value(row["stroke_indices"].clone())?;
        ensure!(!indices.is_empty(), "empty group");
        let mut strokes = Vec::new();
        for &i in &indices {
            ensure!(
                i < n && ownership.insert(i),
                "duplicate/out-of-range native stroke"
            );
            strokes.push(Stroke {
                points: points[offsets[i]..offsets[i + 1]].to_vec(),
                source_index: i,
                uuid: uuid_data[i * 16..(i + 1) * 16]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect(),
                width: f32::from_le_bytes(widths[i * 4..(i + 1) * 4].try_into()?),
            });
        }
        let count: usize = strokes.iter().map(|s| s.points.len()).sum();
        ensure!(
            row["stroke_count"].as_u64() == Some(strokes.len() as u64)
                && row["point_count"].as_u64() == Some(count as u64),
            "group counts mismatch"
        );
        let id = row["sample_id"].as_str().context("sample ID")?;
        let decision = json!({"candidate_id":row["candidate_ids"][0],"source_id":row["source_id"],"status":status,"sample_id":if status == "accepted" {id} else {""},"reason":row["reason"],"worker_id":row["annotator"]});
        decisions.push(decision.clone());
        if status == "unresolved" {
            let mut exclusion = decision;
            exclusion.as_object_mut().unwrap().remove("sample_id");
            exclusions.push(exclusion);
            continue;
        }
        let annotation = json!({"sample_id":id,"source_sha256":source,"candidate_ids":serde_json::to_string(&row["candidate_ids"])?,"original_stroke_indices":serde_json::to_string(&indices)?,"annotator":row["annotator"],"visual_verification":row["visual_verification"]});
        let sample = Sample {
            sample_id: id.into(),
            label: row["latex"].as_str().context("label")?.into(),
            source_sha256: source.into(),
            split: "train".into(),
            strokes,
            metadata: json!({"original":row,"annotation":annotation,"review_status":"supplied-worker-reviewed; independent second pass absent","authorship":"supplied personal dataset; not independently certified"}),
        };
        ensure!(
            serde_json::from_value::<[f32; 4]>(row["bbox"].clone())? == bounds(&sample),
            "bounds mismatch"
        );
        accepted_points += count;
        accepted_strokes += sample.strokes.len();
        samples.push(sample);
        annotations.push(annotation);
    }
    let added = samples.len() - original_count;
    ensure!(ownership.len() == n, "unaccounted source strokes");
    ensure!(
        meta["accepted"].as_u64() == Some(added as u64)
            && meta["unresolved"].as_u64() == Some(exclusions.len() as u64)
            && meta["accepted_points"].as_u64() == Some(accepted_points as u64)
            && meta["accepted_strokes"].as_u64() == Some(accepted_strokes as u64),
        "accepted counts mismatch"
    );
    let report = audit(&samples)?;
    let source_row = json!({"source_id":row_source_id(source),"sha256":source,"filename":meta["source_filename"],"source_url":"user-uploaded-native-archive","archive_format":"GLKeyedArchiver .note ZIP","parse_status":"parsed","archive_size_bytes":meta["source_bytes"],"extracted_stroke_count":n,"extracted_point_count":points.len(),"ink_objects":1,"pen_width_median":median_width(widths)});
    let stats = &mut manifest["source_statistics"];
    for (key, increment) in [
        ("accepted_samples_total", added),
        ("source_archives_total", 1),
        ("source_archives_with_native_ink", 1),
        ("sources_with_accepted_samples", 1),
        ("native_ink_strokes_extracted", n),
        ("native_control_points_extracted", points.len()),
        ("proposed_candidates_total", rows.len()),
        ("candidate_review_count", rows.len()),
        ("total_training_strokes", accepted_strokes),
        ("total_training_control_points", accepted_points),
    ] {
        bump(&mut stats[key], increment)?;
    }
    bump(&mut stats["candidate_status_counts"]["accepted"], added)?;
    bump(
        &mut stats["candidate_status_counts"]["unresolved"],
        exclusions.len(),
    )?;
    bump(
        &mut stats["train_validation_test_sample_counts"]["train"],
        added,
    )?;
    bump(
        &mut stats["train_validation_test_source_counts"]["train"],
        1,
    )?;
    for row in rows.iter().filter(|r| r["status"] == "accepted") {
        bump(
            &mut stats["quality"][row["quality"].as_str().context("quality")?],
            1,
        )?;
        bump(
            &mut stats["content_types"][row["content_type"].as_str().context("content type")?],
            1,
        )?;
    }
    fs::create_dir_all(out)?;
    let mut f = fs::File::create(out.join("corpus.jsonl"))?;
    // Preserve every existing serialized row, including exact held-out bytes.
    f.write_all(&fs::read(corpus.join("corpus.jsonl"))?)?;
    for sample in &samples[original_count..] {
        serde_json::to_writer(&mut f, sample)?;
        writeln!(f)?;
    }
    for (name, extension) in [
        ("sources", vec![source_row]),
        ("annotations", annotations),
        ("candidate_status", decisions),
        ("exclusions", exclusions),
    ] {
        let mut f = fs::File::create(out.join(format!("{name}.jsonl")))?;
        f.write_all(&fs::read(corpus.join(format!("{name}.jsonl")))?)?;
        for row in &extension {
            serde_json::to_writer(&mut f, row)?;
            writeln!(f)?;
        }
        bump(&mut manifest["metadata_table_rows"][name], extension.len())?;
    }
    fs::create_dir(out.join("cache"))?;
    if corpus.join("cache").exists() {
        for entry in fs::read_dir(corpus.join("cache"))? {
            let entry = entry?;
            ensure!(entry.file_type()?.is_file(), "unexpected cache directory");
            fs::copy(entry.path(), out.join("cache").join(entry.file_name()))?;
        }
    }
    fs::write(out.join("cache").join(format!("{source}.npz")), cache)?;
    manifest["audit"] = report;
    manifest["corpus_sha256"] = json!(file_hash(&out.join("corpus.jsonl"))?);
    manifest["supplement_archive_sha256"] = json!(hash(&bytes));
    save(&out.join("manifest.json"), &manifest)?;
    Ok(manifest)
}
fn row_source_id(source: &str) -> &str {
    &source[..20]
}
fn median_width(bytes: &[u8]) -> f32 {
    let mut values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    values.sort_by(f32::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
}
fn bump(value: &mut Value, increment: usize) -> Result<()> {
    *value = json!(
        value
            .as_u64()
            .context("missing numeric statistic")?
            .checked_add(increment as u64)
            .context("count overflow")?
    );
    Ok(())
}

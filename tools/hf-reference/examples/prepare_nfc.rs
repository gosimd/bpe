//! Repository-only preparation. No normalization enters the Go library or timers.
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::{NormalizedString, Normalizer};

fn prepare(input: &Path, output: &Path) -> Result<()> {
    if input.canonicalize()? == output.canonicalize().unwrap_or(output.to_path_buf()) {
        bail!("input and output must differ");
    }
    let mut writer = BufWriter::new(File::create(output)?);
    let mut records = 0;
    for (index, line) in BufReader::new(File::open(input)?).lines().enumerate() {
        let mut record: Value = serde_json::from_str(&line?)?;
        let text = record["text"]
            .as_str()
            .context("record.text must be a string")?;
        let mut normalized = NormalizedString::from(text);
        NFC.normalize(&mut normalized)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        record["text"] = json!(normalized.get());
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
        records = index + 1;
    }
    writer.flush()?;
    if records == 0 {
        bail!("empty corpus");
    }
    println!("prepared {records} NFC records with tokenizers 0.23.2");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        bail!("usage: prepare_nfc INPUT.jsonl OUTPUT.jsonl");
    }
    prepare(Path::new(&args[0]), Path::new(&args[1]))
}

#[test]
fn preserves_empty_records_and_normalizes_independently() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("input.jsonl");
    let output = dir.path().join("output.jsonl");
    std::fs::write(
        &input,
        "{\"text\":\"e\\u0301\"}\n{\"text\":\"\"}\n{\"text\":\"猫\"}\n",
    )?;
    prepare(&input, &output)?;
    assert_eq!(
        std::fs::read_to_string(&output)?,
        "{\"text\":\"é\"}\n{\"text\":\"\"}\n{\"text\":\"猫\"}\n"
    );
    assert!(prepare(&input, &input).is_err());
    Ok(())
}

use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::models::ModelWrapper;
use tokenizers::{
    Encoding, Model, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
};

const TOKENIZERS_VERSION: &str = "0.23.2";
const RUST_TOOLCHAIN: &str = "1.98.1";

#[derive(Parser)]
#[command(version, about = "Pinned Hugging Face tokenizer oracle and benchmark")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a line-oriented reference fixture for every corpus record.
    Generate {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Benchmark the complete Tokenizer::encode pipeline.
    Benchmark {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Human-readable CPU, machine, and execution-environment description.
        #[arg(long)]
        environment: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusRecord {
    text: String,
}

#[derive(Serialize)]
struct FileMetadata {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct Mode {
    add_special_tokens: bool,
    padding: bool,
    truncation: bool,
    offsets: &'static str,
    records: &'static str,
}

impl Default for Mode {
    fn default() -> Self {
        Self {
            add_special_tokens: false,
            padding: false,
            truncation: false,
            offsets: "byte",
            records: "independent",
        }
    }
}

#[derive(Serialize)]
struct FixtureMetadata {
    kind: &'static str,
    schema: &'static str,
    upstream: &'static str,
    tokenizers_version: &'static str,
    rust_toolchain: &'static str,
    tokenizer: FileMetadata,
    corpus: CorpusMetadata,
    mode: Mode,
}

#[derive(Serialize)]
struct CorpusMetadata {
    path: String,
    sha256: String,
    records: usize,
    utf8_bytes: usize,
}

#[derive(Serialize)]
struct FixtureRecord {
    kind: &'static str,
    index: usize,
    input_byte_range: [usize; 2],
    output_token_range: [usize; 2],
    pre_tokenization: Vec<PreTokenPiece>,
    encoding: FinalEncoding,
}

#[derive(Serialize)]
struct PreTokenPiece {
    index: usize,
    value: String,
    offsets: [usize; 2],
    model: PieceModelOutput,
}

#[derive(Serialize)]
struct PieceModelOutput {
    source: &'static str,
    ids: Vec<u32>,
    local_offsets: Vec<[usize; 2]>,
}

#[derive(Serialize)]
struct FinalEncoding {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    offsets: Vec<[usize; 2]>,
    sequence_ids: Vec<Option<usize>>,
    special_tokens_mask: Vec<u32>,
}

struct TracedRecord {
    pieces: Vec<PreTokenPiece>,
    encoding: Encoding,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Aggregate {
    tokens: usize,
    checksum: u64,
}

impl Aggregate {
    fn combine(self, other: Self) -> Self {
        Self {
            tokens: self.tokens + other.tokens,
            checksum: self.checksum.wrapping_add(other.checksum),
        }
    }
}

#[derive(Serialize)]
struct BenchmarkResult {
    schema: &'static str,
    upstream: &'static str,
    tokenizers_version: &'static str,
    rust_toolchain: &'static str,
    operation: &'static str,
    recorded_at_unix_seconds: u64,
    tokenizer: FileMetadata,
    corpus: CorpusMetadata,
    mode: Mode,
    environment: Environment,
    configuration: BenchmarkConfiguration,
    measurement: BenchmarkMeasurement,
}

#[derive(Serialize)]
struct Environment {
    description: String,
    os: &'static str,
    architecture: &'static str,
    available_parallelism: usize,
}

#[derive(Serialize)]
struct BenchmarkConfiguration {
    warmup_iterations: usize,
    measurement_iterations: usize,
    threads: usize,
}

#[derive(Serialize)]
struct BenchmarkMeasurement {
    input_bytes_per_iteration: usize,
    records_per_iteration: usize,
    tokens_per_iteration: usize,
    checksum: u64,
    iteration_nanoseconds: Vec<u128>,
    min_nanoseconds: u128,
    median_nanoseconds: u128,
    max_nanoseconds: u128,
    median_nanoseconds_per_input_byte: f64,
    median_mib_per_second: f64,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Generate {
            tokenizer,
            corpus,
            output,
        } => generate_fixture(&tokenizer, &corpus, &output),
        Command::Benchmark {
            tokenizer,
            corpus,
            output,
            warmup,
            iterations,
            threads,
            environment,
        } => run_benchmark(
            &tokenizer,
            &corpus,
            &output,
            warmup,
            iterations,
            threads,
            environment,
        ),
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("load tokenizer from {}", path.display()))?;
    tokenizer
        .with_truncation(None)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    tokenizer.with_padding(None);
    if !matches!(tokenizer.get_model(), ModelWrapper::BPE(_)) {
        bail!(
            "{} does not contain a BPE model; the three-level oracle is BPE-specific",
            path.display()
        );
    }
    Ok(tokenizer)
}

fn load_corpus(path: &Path) -> Result<Vec<CorpusRecord>> {
    let file = File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    let mut records = Vec::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
        if line.is_empty() {
            bail!(
                "{} line {} is physically empty; encode an empty record as {{\"text\":\"\"}}",
                path.display(),
                index + 1
            );
        }
        let record = serde_json::from_str(&line)
            .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
        records.push(record);
    }

    if records.is_empty() {
        bail!("corpus {} contains no records", path.display());
    }
    Ok(records)
}

fn corpus_metadata(path: &Path, records: &[CorpusRecord]) -> Result<CorpusMetadata> {
    Ok(CorpusMetadata {
        path: path.display().to_string(),
        sha256: sha256(path)?,
        records: records.len(),
        utf8_bytes: records.iter().map(|record| record.text.len()).sum(),
    })
}

fn generate_fixture(tokenizer_path: &Path, corpus_path: &Path, output: &Path) -> Result<()> {
    reject_input_overwrite(tokenizer_path, corpus_path, output)?;
    let tokenizer = load_tokenizer(tokenizer_path)?;
    let records = load_corpus(corpus_path)?;
    let metadata = FixtureMetadata {
        kind: "metadata",
        schema: "gosimd-bpe-hf-fixture-v2",
        upstream: "huggingface/tokenizers",
        tokenizers_version: TOKENIZERS_VERSION,
        rust_toolchain: RUST_TOOLCHAIN,
        tokenizer: FileMetadata {
            path: tokenizer_path.display().to_string(),
            sha256: sha256(tokenizer_path)?,
        },
        corpus: corpus_metadata(corpus_path, &records)?,
        mode: Mode::default(),
    };

    ensure_parent(output)?;
    let file = File::create(output).with_context(|| format!("create {}", output.display()))?;
    let mut writer = BufWriter::new(file);
    write_json_line(&mut writer, &metadata)?;

    let mut output_token_offset = 0;
    let mut total_pieces = 0;
    for (index, record) in records.iter().enumerate() {
        let traced = trace_record(&tokenizer, &record.text)
            .with_context(|| format!("trace corpus record {index}"))?;
        let output_token_end = output_token_offset + traced.encoding.len();
        total_pieces += traced.pieces.len();
        let fixture = FixtureRecord {
            kind: "record",
            index,
            input_byte_range: [0, record.text.len()],
            output_token_range: [output_token_offset, output_token_end],
            pre_tokenization: traced.pieces,
            encoding: snapshot_encoding(&traced.encoding),
        };
        write_json_line(&mut writer, &fixture)?;
        output_token_offset = output_token_end;
    }

    writer
        .flush()
        .with_context(|| format!("flush {}", output.display()))?;
    println!(
        "wrote {} records, {} pieces, and {} tokens to {}",
        records.len(),
        total_pieces,
        output_token_offset,
        output.display()
    );
    Ok(())
}

fn trace_record(tokenizer: &Tokenizer, text: &str) -> Result<TracedRecord> {
    let mut pretokenized: PreTokenizedString = tokenizer
        .get_added_vocabulary()
        .extract_and_normalize(tokenizer.get_normalizer(), text);
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        pre_tokenizer
            .pre_tokenize(&mut pretokenized)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }

    let sources = pretokenized
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(_, _, tokens)| {
            if tokens.is_some() {
                "added_token"
            } else {
                "bpe"
            }
        })
        .collect::<Vec<_>>();

    tokenizer
        .get_model()
        .tokenize_in_pretokenized(&mut pretokenized, None)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
    if splits.len() != sources.len() {
        bail!("pre-tokenized split count changed during model tokenization");
    }
    let pieces = splits
        .into_iter()
        .zip(sources)
        .enumerate()
        .map(|(index, ((value, (start, end), tokens), source))| {
            let tokens = tokens
                .as_ref()
                .with_context(|| format!("piece {index} has no model tokens"))?;
            Ok(PreTokenPiece {
                index,
                value: value.to_owned(),
                offsets: [start, end],
                model: PieceModelOutput {
                    source,
                    ids: tokens.iter().map(|token| token.id).collect(),
                    local_offsets: tokens
                        .iter()
                        .map(|token| [token.offsets.0, token.offsets.1])
                        .collect(),
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let model_encoding = pretokenized
        .into_encoding(None, 0, OffsetType::Byte)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let reconstructed = tokenizer
        .post_process(model_encoding, None, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let encoding = encode(tokenizer, text)?;
    if reconstructed != encoding {
        bail!("manually traced pipeline differs from Tokenizer::encode");
    }

    Ok(TracedRecord { pieces, encoding })
}

fn snapshot_encoding(encoding: &Encoding) -> FinalEncoding {
    FinalEncoding {
        ids: encoding.get_ids().to_vec(),
        type_ids: encoding.get_type_ids().to_vec(),
        offsets: encoding
            .get_offsets()
            .iter()
            .map(|&(start, end)| [start, end])
            .collect(),
        sequence_ids: encoding.get_sequence_ids(),
        special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
    }
}

fn run_benchmark(
    tokenizer_path: &Path,
    corpus_path: &Path,
    output: &Path,
    warmup: usize,
    iterations: usize,
    threads: usize,
    environment_description: String,
) -> Result<()> {
    reject_input_overwrite(tokenizer_path, corpus_path, output)?;
    if iterations == 0 {
        bail!("--iterations must be greater than zero");
    }
    if threads == 0 {
        bail!("--threads must be greater than zero");
    }
    if environment_description.trim().is_empty() {
        bail!("--environment must not be empty");
    }

    let tokenizer = load_tokenizer(tokenizer_path)?;
    let records = load_corpus(corpus_path)?;
    let pool = if threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("hf-reference-{index}"))
                .build()
                .context("create benchmark thread pool")?,
        )
    } else {
        None
    };

    for _ in 0..warmup {
        black_box(encode_corpus(&tokenizer, &records, pool.as_ref())?);
    }

    let mut durations = Vec::with_capacity(iterations);
    let mut expected = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let aggregate = black_box(encode_corpus(&tokenizer, &records, pool.as_ref())?);
        let elapsed = started.elapsed().as_nanos();
        if let Some(expected) = expected {
            if aggregate != expected {
                bail!("token count or checksum changed between benchmark iterations");
            }
        } else {
            expected = Some(aggregate);
        }
        durations.push(elapsed);
    }

    let aggregate = expected.expect("iterations is non-zero");
    let input_bytes = records
        .iter()
        .map(|record| record.text.len())
        .sum::<usize>();
    if input_bytes == 0 {
        bail!("cannot benchmark a corpus containing zero UTF-8 bytes");
    }
    let median = median(&durations);
    let seconds = median as f64 / 1_000_000_000.0;
    let result = BenchmarkResult {
        schema: "gosimd-bpe-hf-benchmark-v1",
        upstream: "huggingface/tokenizers",
        tokenizers_version: TOKENIZERS_VERSION,
        rust_toolchain: RUST_TOOLCHAIN,
        operation: "Tokenizer::encode/full_pipeline",
        recorded_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_secs(),
        tokenizer: FileMetadata {
            path: tokenizer_path.display().to_string(),
            sha256: sha256(tokenizer_path)?,
        },
        corpus: corpus_metadata(corpus_path, &records)?,
        mode: Mode::default(),
        environment: Environment {
            description: environment_description,
            os: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            available_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        },
        configuration: BenchmarkConfiguration {
            warmup_iterations: warmup,
            measurement_iterations: iterations,
            threads,
        },
        measurement: BenchmarkMeasurement {
            input_bytes_per_iteration: input_bytes,
            records_per_iteration: records.len(),
            tokens_per_iteration: aggregate.tokens,
            checksum: aggregate.checksum,
            iteration_nanoseconds: durations.clone(),
            min_nanoseconds: *durations.iter().min().expect("durations is non-empty"),
            median_nanoseconds: median,
            max_nanoseconds: *durations.iter().max().expect("durations is non-empty"),
            median_nanoseconds_per_input_byte: median as f64 / input_bytes as f64,
            median_mib_per_second: input_bytes as f64 / (1024.0 * 1024.0) / seconds,
        },
    };

    ensure_parent(output)?;
    let mut file = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut file, &result)
        .with_context(|| format!("serialize {}", output.display()))?;
    file.write_all(b"\n")?;
    file.flush()?;
    println!("wrote benchmark result to {}", output.display());
    Ok(())
}

fn encode_corpus(
    tokenizer: &Tokenizer,
    records: &[CorpusRecord],
    pool: Option<&ThreadPool>,
) -> Result<Aggregate> {
    if let Some(pool) = pool {
        pool.install(|| {
            records
                .par_iter()
                .map(|record| encode(tokenizer, &record.text).map(|encoding| digest(&encoding)))
                .try_reduce(Aggregate::default, |left, right| Ok(left.combine(right)))
        })
    } else {
        records
            .iter()
            .try_fold(Aggregate::default(), |total, record| {
                let encoding = encode(tokenizer, &record.text)?;
                Ok(total.combine(digest(&encoding)))
            })
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Encoding> {
    tokenizer
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn digest(encoding: &Encoding) -> Aggregate {
    let mut checksum = 0xcbf29ce484222325_u64;
    for &id in encoding.get_ids() {
        checksum ^= u64::from(id);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    Aggregate {
        tokens: encoding.len(),
        checksum,
    }
}

fn median(values: &[u128]) -> u128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

fn reject_input_overwrite(tokenizer: &Path, corpus: &Path, output: &Path) -> Result<()> {
    if output == tokenizer || output == corpus {
        bail!("output path must differ from tokenizer and corpus input paths");
    }
    Ok(())
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_loader_preserves_empty_records() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("corpus.jsonl");
        fs::write(&path, "{\"text\":\"first\"}\n{\"text\":\"\"}\n").unwrap();

        let records = load_corpus(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].text, "first");
        assert_eq!(records[1].text, "");
    }

    #[test]
    fn corpus_loader_rejects_physical_blank_lines() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("corpus.jsonl");
        fs::write(&path, "{\"text\":\"first\"}\n\n").unwrap();

        let error = load_corpus(&path).unwrap_err();
        assert!(error.to_string().contains("physically empty"));
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&[9, 1, 5]), 5);
        assert_eq!(median(&[10, 2, 8, 4]), 6);
    }

    #[test]
    fn trace_exposes_all_gpt2_pipeline_levels() {
        let tokenizer_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/gpt2/tokenizer.json");
        let tokenizer = load_tokenizer(&tokenizer_path).unwrap();

        let traced = trace_record(&tokenizer, "Hello, world!").unwrap();
        assert_eq!(
            traced
                .pieces
                .iter()
                .map(|piece| piece.value.as_str())
                .collect::<Vec<_>>(),
            ["Hello", ",", "Ġworld", "!"]
        );
        assert_eq!(
            traced
                .pieces
                .iter()
                .map(|piece| piece.offsets)
                .collect::<Vec<_>>(),
            [[0, 5], [5, 6], [6, 12], [12, 13]]
        );
        assert!(traced
            .pieces
            .iter()
            .all(|piece| piece.model.source == "bpe"));
        assert_eq!(traced.pieces[2].model.ids, [995]);
        assert_eq!(traced.pieces[2].model.local_offsets, [[0, 7]]);
        assert_eq!(traced.encoding.get_ids(), [15496, 11, 995, 0]);
        assert_eq!(
            traced.encoding.get_offsets(),
            [(0, 5), (5, 6), (6, 12), (12, 13)]
        );

        let added = trace_record(&tokenizer, "<|endoftext|>").unwrap();
        assert_eq!(added.pieces.len(), 1);
        assert_eq!(added.pieces[0].value, "<|endoftext|>");
        assert_eq!(added.pieces[0].offsets, [0, 13]);
        assert_eq!(added.pieces[0].model.source, "added_token");
        assert_eq!(added.pieces[0].model.ids, [50256]);
        assert_eq!(added.pieces[0].model.local_offsets, [[0, 13]]);
        assert_eq!(added.encoding.get_ids(), [50256]);
    }
}

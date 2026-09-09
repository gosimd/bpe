use std::collections::HashSet;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use tokenizers::models::{bpe::BPE, ModelWrapper};
use tokenizers::{Model, PreTokenizedString, PreTokenizer, Tokenizer};

use super::{
    corpus_metadata, encode, ensure_parent, load_corpus, load_tokenizer, sha256, Aggregate,
    CorpusMetadata, CorpusRecord, FileMetadata, Mode, RUST_TOOLCHAIN, TOKENIZERS_VERSION,
};

const FIXTURE_SCHEMA: &str = "gosimd-bpe-hf-fixture-v2";
const DEFAULT_BPE_CACHE_CAPACITY: usize = 10_000;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum BenchmarkOperation {
    PretokenizeOnly,
    BpeOnlyUncached,
    EncodeFullUncached,
    EncodeFullWarmCache,
}

impl BenchmarkOperation {
    fn name(self) -> &'static str {
        match self {
            Self::PretokenizeOnly => "pretokenize_only",
            Self::BpeOnlyUncached => "bpe_only_uncached",
            Self::EncodeFullUncached => "encode_full_uncached",
            Self::EncodeFullWarmCache => "encode_full_warm_cache",
        }
    }
}

#[derive(Serialize)]
struct BenchmarkResult {
    schema: &'static str,
    upstream: &'static str,
    tokenizers_version: &'static str,
    rust_toolchain: &'static str,
    tool: ToolMetadata,
    recorded_at_unix_seconds: u64,
    tokenizer: FileMetadata,
    vocab: FileMetadata,
    merges: FileMetadata,
    corpus: CorpusMetadata,
    fixture: FileMetadata,
    mode: Mode,
    workload: Workload,
    environment: Environment,
    configuration: BenchmarkConfiguration,
    measurements: Vec<BenchmarkMeasurement>,
}

#[derive(Serialize)]
struct ToolMetadata {
    package_version: &'static str,
    main_source: FileMetadata,
    benchmark_source: FileMetadata,
    cargo_lock: FileMetadata,
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
    timing_scope: &'static str,
}

#[derive(Serialize)]
struct Workload {
    pre_tokenized_pieces_per_iteration: usize,
    bpe_model_pieces_per_iteration: usize,
    bpe_cache_eligible_unique_pieces: usize,
    bpe_default_cache_capacity: usize,
}

#[derive(Serialize)]
struct BenchmarkMeasurement {
    operation: &'static str,
    cache: &'static str,
    cache_capacity: Option<usize>,
    source_records_per_iteration: usize,
    source_utf8_bytes_per_iteration: usize,
    operation_items_per_iteration: usize,
    operation_input_bytes_per_iteration: usize,
    pieces_per_iteration: usize,
    tokens_per_iteration: Option<usize>,
    oracle_checksum: u64,
    iteration_nanoseconds: Vec<u128>,
    min_nanoseconds: u128,
    median_nanoseconds: u128,
    max_nanoseconds: u128,
    median_nanoseconds_per_operation_input_byte: f64,
    median_operation_mib_per_second: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleMetadata {
    kind: String,
    schema: String,
    upstream: String,
    tokenizers_version: String,
    rust_toolchain: String,
    tokenizer: FileMetadata,
    corpus: CorpusMetadata,
    mode: OracleMode,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleMode {
    add_special_tokens: bool,
    padding: bool,
    truncation: bool,
    offsets: String,
    records: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleRecord {
    kind: String,
    index: usize,
    input_byte_range: [usize; 2],
    output_token_range: [usize; 2],
    pre_tokenization: Vec<OraclePiece>,
    encoding: OracleEncoding,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OraclePiece {
    index: usize,
    value: String,
    offsets: [usize; 2],
    model: OracleModelOutput,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleModelOutput {
    source: String,
    ids: Vec<u32>,
    local_offsets: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleEncoding {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    offsets: Vec<[usize; 2]>,
    sequence_ids: Vec<Option<usize>>,
    special_tokens_mask: Vec<u32>,
}

struct PreparedPiece {
    value: String,
    ids: Vec<u32>,
}

struct PreparedFixture {
    all_pieces: usize,
    model_pieces: Vec<PreparedPiece>,
    model_input_bytes: usize,
    model_tokens: usize,
    model_checksum: u64,
    cache_eligible_unique_pieces: usize,
    record_ids: Vec<Vec<u32>>,
    encoding_tokens: usize,
    encoding_checksum: u64,
    piece_checksum: u64,
}

struct MeasurementShape {
    operation: BenchmarkOperation,
    cache: &'static str,
    cache_capacity: Option<usize>,
    source_records: usize,
    source_bytes: usize,
    operation_items: usize,
    operation_bytes: usize,
    pieces: usize,
    tokens: Option<usize>,
    checksum: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_benchmark(
    tokenizer_path: &Path,
    corpus_path: &Path,
    vocab_path: &Path,
    merges_path: &Path,
    fixture_path: &Path,
    output: &Path,
    warmup: usize,
    iterations: usize,
    threads: usize,
    environment_description: String,
) -> Result<()> {
    reject_benchmark_output_overwrite(
        tokenizer_path,
        corpus_path,
        vocab_path,
        merges_path,
        fixture_path,
        output,
    )?;
    validate_run_options(iterations, threads, &environment_description)?;

    let records = load_corpus(corpus_path)?;
    let prepared = load_fixture(fixture_path, tokenizer_path, corpus_path, &records)?;
    let pool = build_pool(threads)?;

    let mut uncached_tokenizer = load_tokenizer(tokenizer_path)?;
    let uncached_model = build_uncached_model(&uncached_tokenizer, vocab_path, merges_path)?;
    uncached_tokenizer.with_model(uncached_model);
    let warm_tokenizer = load_tokenizer(tokenizer_path)?;

    validate_pre_tokenization(&uncached_tokenizer, &records, prepared.all_pieces)?;
    validate_bpe(uncached_tokenizer.get_model(), &prepared.model_pieces)?;
    validate_full_encoding(&uncached_tokenizer, &records, &prepared.record_ids)?;

    // Populate the configured thread-local cache on the same thread or Rayon
    // pool that will execute the warm-cache measurements.
    black_box(encode_corpus(&warm_tokenizer, &records, pool.as_ref())?);

    let source_bytes = records.iter().map(|record| record.text.len()).sum();
    let common = |operation,
                  cache,
                  cache_capacity,
                  operation_items,
                  operation_bytes,
                  pieces,
                  tokens,
                  checksum| {
        MeasurementShape {
            operation,
            cache,
            cache_capacity,
            source_records: records.len(),
            source_bytes,
            operation_items,
            operation_bytes,
            pieces,
            tokens,
            checksum,
        }
    };

    let measurements = vec![
        measure(
            common(
                BenchmarkOperation::PretokenizeOnly,
                "not_applicable",
                None,
                records.len(),
                source_bytes,
                prepared.all_pieces,
                None,
                prepared.piece_checksum,
            ),
            warmup,
            iterations,
            || pretokenize_corpus(&uncached_tokenizer, &records, pool.as_ref()),
        )?,
        measure(
            common(
                BenchmarkOperation::BpeOnlyUncached,
                "disabled_at_model_construction",
                Some(0),
                prepared.model_pieces.len(),
                prepared.model_input_bytes,
                prepared.model_pieces.len(),
                Some(prepared.model_tokens),
                prepared.model_checksum,
            ),
            warmup,
            iterations,
            || {
                tokenize_pieces(
                    uncached_tokenizer.get_model(),
                    &prepared.model_pieces,
                    pool.as_ref(),
                )
            },
        )?,
        measure(
            common(
                BenchmarkOperation::EncodeFullUncached,
                "disabled_at_model_construction",
                Some(0),
                records.len(),
                source_bytes,
                prepared.all_pieces,
                Some(prepared.encoding_tokens),
                prepared.encoding_checksum,
            ),
            warmup,
            iterations,
            || encode_corpus(&uncached_tokenizer, &records, pool.as_ref()),
        )?,
        measure(
            common(
                BenchmarkOperation::EncodeFullWarmCache,
                "default_thread_local_cache_prefilled_once",
                Some(DEFAULT_BPE_CACHE_CAPACITY),
                records.len(),
                source_bytes,
                prepared.all_pieces,
                Some(prepared.encoding_tokens),
                prepared.encoding_checksum,
            ),
            warmup,
            iterations,
            || encode_corpus(&warm_tokenizer, &records, pool.as_ref()),
        )?,
    ];

    let result = BenchmarkResult {
        schema: "gosimd-bpe-hf-benchmark-v2",
        upstream: "huggingface/tokenizers",
        tokenizers_version: TOKENIZERS_VERSION,
        rust_toolchain: RUST_TOOLCHAIN,
        tool: tool_metadata()?,
        recorded_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_secs(),
        tokenizer: file_metadata(tokenizer_path)?,
        vocab: file_metadata(vocab_path)?,
        merges: file_metadata(merges_path)?,
        corpus: corpus_metadata(corpus_path, &records)?,
        fixture: file_metadata(fixture_path)?,
        mode: Mode::default(),
        workload: Workload {
            pre_tokenized_pieces_per_iteration: prepared.all_pieces,
            bpe_model_pieces_per_iteration: prepared.model_pieces.len(),
            bpe_cache_eligible_unique_pieces: prepared.cache_eligible_unique_pieces,
            bpe_default_cache_capacity: DEFAULT_BPE_CACHE_CAPACITY,
        },
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
            timing_scope: "operation plus output construction/destruction; input loading, fixture validation, model construction, and cache prefill excluded",
        },
        measurements,
    };

    ensure_parent(output)?;
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &result)
        .with_context(|| format!("serialize {}", output.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("wrote benchmark result to {}", output.display());
    Ok(())
}

fn tool_metadata() -> Result<ToolMetadata> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    Ok(ToolMetadata {
        package_version: env!("CARGO_PKG_VERSION"),
        main_source: file_metadata_named(
            "tools/hf-reference/src/main.rs",
            &manifest.join("src/main.rs"),
        )?,
        benchmark_source: file_metadata_named(
            "tools/hf-reference/src/benchmark.rs",
            &manifest.join("src/benchmark.rs"),
        )?,
        cargo_lock: file_metadata_named(
            "tools/hf-reference/Cargo.lock",
            &manifest.join("Cargo.lock"),
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_profile(
    tokenizer_path: &Path,
    corpus_path: &Path,
    vocab_path: &Path,
    merges_path: &Path,
    fixture_path: &Path,
    operation: BenchmarkOperation,
    seconds: u64,
    threads: usize,
) -> Result<()> {
    if seconds == 0 {
        bail!("--seconds must be greater than zero");
    }
    if threads == 0 {
        bail!("--threads must be greater than zero");
    }

    let records = load_corpus(corpus_path)?;
    let prepared = load_fixture(fixture_path, tokenizer_path, corpus_path, &records)?;
    let pool = build_pool(threads)?;
    let mut uncached_tokenizer = load_tokenizer(tokenizer_path)?;
    let uncached_model = build_uncached_model(&uncached_tokenizer, vocab_path, merges_path)?;
    uncached_tokenizer.with_model(uncached_model);
    let warm_tokenizer = load_tokenizer(tokenizer_path)?;

    validate_pre_tokenization(&uncached_tokenizer, &records, prepared.all_pieces)?;
    validate_bpe(uncached_tokenizer.get_model(), &prepared.model_pieces)?;
    validate_full_encoding(&uncached_tokenizer, &records, &prepared.record_ids)?;
    if matches!(operation, BenchmarkOperation::EncodeFullWarmCache) {
        black_box(encode_corpus(&warm_tokenizer, &records, pool.as_ref())?);
    }

    println!(
        "PROFILE_READY operation={} seconds={} threads={}",
        operation.name(),
        seconds,
        threads
    );
    std::io::stdout().flush()?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut passes = 0_u64;
    let mut marker = Aggregate::default();
    while Instant::now() < deadline {
        marker = match operation {
            BenchmarkOperation::PretokenizeOnly => {
                pretokenize_corpus(&uncached_tokenizer, &records, pool.as_ref())?
            }
            BenchmarkOperation::BpeOnlyUncached => tokenize_pieces(
                uncached_tokenizer.get_model(),
                &prepared.model_pieces,
                pool.as_ref(),
            )?,
            BenchmarkOperation::EncodeFullUncached => {
                encode_corpus(&uncached_tokenizer, &records, pool.as_ref())?
            }
            BenchmarkOperation::EncodeFullWarmCache => {
                encode_corpus(&warm_tokenizer, &records, pool.as_ref())?
            }
        };
        black_box(marker);
        passes += 1;
    }
    println!(
        "PROFILE_COMPLETE operation={} passes={} marker={}:{}",
        operation.name(),
        passes,
        marker.tokens,
        marker.checksum
    );
    Ok(())
}

fn validate_run_options(iterations: usize, threads: usize, environment: &str) -> Result<()> {
    if iterations == 0 {
        bail!("--iterations must be greater than zero");
    }
    if threads == 0 {
        bail!("--threads must be greater than zero");
    }
    if environment.trim().is_empty() {
        bail!("--environment must not be empty");
    }
    Ok(())
}

fn build_pool(threads: usize) -> Result<Option<ThreadPool>> {
    if threads == 1 {
        return Ok(None);
    }
    Ok(Some(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("hf-reference-{index}"))
            .build()
            .context("create benchmark thread pool")?,
    ))
}

fn build_uncached_model(
    tokenizer: &Tokenizer,
    vocab_path: &Path,
    merges_path: &Path,
) -> Result<ModelWrapper> {
    let current = match tokenizer.get_model() {
        ModelWrapper::BPE(model) => model,
        _ => bail!("uncached model construction requires BPE"),
    };
    let vocab = vocab_path
        .to_str()
        .with_context(|| format!("vocabulary path is not UTF-8: {}", vocab_path.display()))?;
    let merges = merges_path
        .to_str()
        .with_context(|| format!("merges path is not UTF-8: {}", merges_path.display()))?;
    let mut builder = BPE::from_file(vocab, merges)
        .cache_capacity(0)
        .fuse_unk(current.fuse_unk)
        .byte_fallback(current.byte_fallback)
        .ignore_merges(current.ignore_merges);
    if let Some(dropout) = current.dropout {
        builder = builder.dropout(dropout);
    }
    if let Some(token) = &current.unk_token {
        builder = builder.unk_token(token.clone());
    }
    if let Some(prefix) = &current.continuing_subword_prefix {
        builder = builder.continuing_subword_prefix(prefix.clone());
    }
    if let Some(suffix) = &current.end_of_word_suffix {
        builder = builder.end_of_word_suffix(suffix.clone());
    }
    Ok(ModelWrapper::BPE(
        builder
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?,
    ))
}

fn pretokenize(tokenizer: &Tokenizer, text: &str) -> Result<PreTokenizedString> {
    let mut pretokenized: PreTokenizedString = tokenizer
        .get_added_vocabulary()
        .extract_and_normalize(tokenizer.get_normalizer(), text);
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        pre_tokenizer
            .pre_tokenize(&mut pretokenized)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    Ok(pretokenized)
}

fn pretokenize_corpus(
    tokenizer: &Tokenizer,
    records: &[CorpusRecord],
    pool: Option<&ThreadPool>,
) -> Result<Aggregate> {
    if let Some(pool) = pool {
        pool.install(|| {
            records
                .par_iter()
                .map(|record| {
                    black_box(pretokenize(tokenizer, &record.text)?);
                    Ok(Aggregate {
                        tokens: 1,
                        checksum: record.text.len() as u64,
                    })
                })
                .try_reduce(Aggregate::default, |left, right| Ok(left.combine(right)))
        })
    } else {
        records
            .iter()
            .try_fold(Aggregate::default(), |total, record| {
                black_box(pretokenize(tokenizer, &record.text)?);
                Ok(total.combine(Aggregate {
                    tokens: 1,
                    checksum: record.text.len() as u64,
                }))
            })
    }
}

fn tokenize_pieces(
    model: &ModelWrapper,
    pieces: &[PreparedPiece],
    pool: Option<&ThreadPool>,
) -> Result<Aggregate> {
    if let Some(pool) = pool {
        pool.install(|| {
            pieces
                .par_iter()
                .map(|piece| {
                    let tokens = model
                        .tokenize(&piece.value)
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                    black_box(tokens);
                    Ok(Aggregate {
                        tokens: 1,
                        checksum: piece.value.len() as u64,
                    })
                })
                .try_reduce(Aggregate::default, |left, right| Ok(left.combine(right)))
        })
    } else {
        pieces
            .iter()
            .try_fold(Aggregate::default(), |total, piece| {
                let tokens = model
                    .tokenize(&piece.value)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                black_box(tokens);
                Ok(total.combine(Aggregate {
                    tokens: 1,
                    checksum: piece.value.len() as u64,
                }))
            })
    }
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
                .map(|record| {
                    black_box(encode(tokenizer, &record.text)?);
                    Ok(Aggregate {
                        tokens: 1,
                        checksum: record.text.len() as u64,
                    })
                })
                .try_reduce(Aggregate::default, |left, right| Ok(left.combine(right)))
        })
    } else {
        records
            .iter()
            .try_fold(Aggregate::default(), |total, record| {
                black_box(encode(tokenizer, &record.text)?);
                Ok(total.combine(Aggregate {
                    tokens: 1,
                    checksum: record.text.len() as u64,
                }))
            })
    }
}

fn measure(
    shape: MeasurementShape,
    warmup: usize,
    iterations: usize,
    mut run: impl FnMut() -> Result<Aggregate>,
) -> Result<BenchmarkMeasurement> {
    for _ in 0..warmup {
        black_box(run()?);
    }

    let mut marker = None;
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let current = black_box(run()?);
        durations.push(started.elapsed().as_nanos());
        if let Some(expected) = marker {
            if current != expected {
                bail!(
                    "{} marker changed between iterations",
                    shape.operation.name()
                );
            }
        } else {
            marker = Some(current);
        }
    }

    let median = median(&durations);
    let seconds = median as f64 / 1_000_000_000.0;
    let measurement = BenchmarkMeasurement {
        operation: shape.operation.name(),
        cache: shape.cache,
        cache_capacity: shape.cache_capacity,
        source_records_per_iteration: shape.source_records,
        source_utf8_bytes_per_iteration: shape.source_bytes,
        operation_items_per_iteration: shape.operation_items,
        operation_input_bytes_per_iteration: shape.operation_bytes,
        pieces_per_iteration: shape.pieces,
        tokens_per_iteration: shape.tokens,
        oracle_checksum: shape.checksum,
        iteration_nanoseconds: durations.clone(),
        min_nanoseconds: *durations.iter().min().expect("durations is non-empty"),
        median_nanoseconds: median,
        max_nanoseconds: *durations.iter().max().expect("durations is non-empty"),
        median_nanoseconds_per_operation_input_byte: median as f64 / shape.operation_bytes as f64,
        median_operation_mib_per_second: shape.operation_bytes as f64 / (1024.0 * 1024.0) / seconds,
    };
    println!(
        "{}: median {:.3} ms, {:.2} MiB/s",
        measurement.operation,
        measurement.median_nanoseconds as f64 / 1_000_000.0,
        measurement.median_operation_mib_per_second
    );
    Ok(measurement)
}

fn validate_pre_tokenization(
    tokenizer: &Tokenizer,
    records: &[CorpusRecord],
    want_pieces: usize,
) -> Result<()> {
    let mut pieces = 0;
    for record in records {
        pieces += pretokenize(tokenizer, &record.text)?
            .get_splits(
                tokenizers::OffsetReferential::Original,
                tokenizers::OffsetType::Byte,
            )
            .len();
    }
    if pieces != want_pieces {
        bail!("prepared fixture has {want_pieces} pieces, tokenizer produced {pieces}");
    }
    Ok(())
}

fn validate_bpe(model: &ModelWrapper, pieces: &[PreparedPiece]) -> Result<()> {
    for (index, piece) in pieces.iter().enumerate() {
        let tokens = model
            .tokenize(&piece.value)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let ids = tokens.iter().map(|token| token.id).collect::<Vec<_>>();
        if ids != piece.ids {
            bail!("uncached BPE differs from fixture at model piece {index}");
        }
    }
    Ok(())
}

fn validate_full_encoding(
    tokenizer: &Tokenizer,
    records: &[CorpusRecord],
    want_ids: &[Vec<u32>],
) -> Result<()> {
    if records.len() != want_ids.len() {
        bail!("corpus and fixture record counts differ");
    }
    for (index, (record, want)) in records.iter().zip(want_ids).enumerate() {
        let encoding = encode(tokenizer, &record.text)?;
        if encoding.get_ids() != want {
            bail!("uncached full encoding differs from fixture at record {index}");
        }
    }
    Ok(())
}

fn load_fixture(
    path: &Path,
    tokenizer_path: &Path,
    corpus_path: &Path,
    records: &[CorpusRecord],
) -> Result<PreparedFixture> {
    let file = File::open(path).with_context(|| format!("open fixture {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let metadata_line = lines
        .next()
        .context("fixture is empty")?
        .with_context(|| format!("read fixture metadata from {}", path.display()))?;
    let metadata: OracleMetadata = serde_json::from_str(&metadata_line)
        .with_context(|| format!("parse fixture metadata from {}", path.display()))?;
    validate_fixture_metadata(&metadata, tokenizer_path, corpus_path, records)?;

    let mut all_pieces = 0;
    let mut model_pieces = Vec::new();
    let mut model_input_bytes = 0;
    let mut model_aggregate = Aggregate::default();
    let mut cache_eligible = HashSet::new();
    let mut encoding_aggregate = Aggregate::default();
    let mut record_ids = Vec::with_capacity(records.len());
    let mut piece_checksum = 0xcbf29ce484222325_u64;
    let mut output_position = 0;

    for (index, line) in lines.enumerate() {
        let line = line.with_context(|| format!("read fixture record {index}"))?;
        let record: OracleRecord =
            serde_json::from_str(&line).with_context(|| format!("parse fixture record {index}"))?;
        if record.kind != "record" || record.index != index {
            bail!(
                "fixture record {index} has kind {:?} and index {}",
                record.kind,
                record.index
            );
        }
        if index >= records.len() {
            bail!("fixture has more records than corpus");
        }
        if record.input_byte_range != [0, records[index].text.len()] {
            bail!("fixture input byte range differs at record {index}");
        }
        if record.output_token_range
            != [output_position, output_position + record.encoding.ids.len()]
        {
            bail!("fixture output token range differs at record {index}");
        }
        validate_encoding_shape(index, &record.encoding)?;

        for (piece_index, piece) in record.pre_tokenization.into_iter().enumerate() {
            if piece.index != piece_index || piece.offsets[0] > piece.offsets[1] {
                bail!("invalid fixture piece {piece_index} at record {index}");
            }
            if piece.model.ids.len() != piece.model.local_offsets.len() {
                bail!("fixture piece {piece_index} has mismatched IDs and offsets");
            }
            all_pieces += 1;
            piece_checksum = checksum_bytes(piece_checksum, piece.value.as_bytes());
            match piece.model.source.as_str() {
                "bpe" => {
                    model_input_bytes += piece.value.len();
                    model_aggregate = model_aggregate.combine(digest_ids(&piece.model.ids));
                    if piece.value.len() < 256 {
                        cache_eligible.insert(piece.value.clone());
                    }
                    model_pieces.push(PreparedPiece {
                        value: piece.value,
                        ids: piece.model.ids,
                    });
                }
                "added_token" => {}
                source => bail!("unknown model source {source:?} at record {index}"),
            }
        }
        output_position += record.encoding.ids.len();
        encoding_aggregate = encoding_aggregate.combine(digest_ids(&record.encoding.ids));
        record_ids.push(record.encoding.ids);
    }
    if record_ids.len() != records.len() {
        bail!(
            "fixture has {} records, corpus has {}",
            record_ids.len(),
            records.len()
        );
    }

    Ok(PreparedFixture {
        all_pieces,
        model_input_bytes,
        model_tokens: model_aggregate.tokens,
        model_checksum: model_aggregate.checksum,
        cache_eligible_unique_pieces: cache_eligible.len(),
        model_pieces,
        encoding_tokens: encoding_aggregate.tokens,
        encoding_checksum: encoding_aggregate.checksum,
        record_ids,
        piece_checksum,
    })
}

fn validate_fixture_metadata(
    metadata: &OracleMetadata,
    tokenizer_path: &Path,
    corpus_path: &Path,
    records: &[CorpusRecord],
) -> Result<()> {
    if metadata.kind != "metadata"
        || metadata.schema != FIXTURE_SCHEMA
        || metadata.upstream != "huggingface/tokenizers"
        || metadata.tokenizers_version != TOKENIZERS_VERSION
        || metadata.rust_toolchain != RUST_TOOLCHAIN
    {
        bail!("fixture metadata does not match the pinned reference configuration");
    }
    if metadata.mode.add_special_tokens
        || metadata.mode.padding
        || metadata.mode.truncation
        || metadata.mode.offsets != "byte"
        || metadata.mode.records != "independent"
    {
        bail!("fixture encoding mode is unsupported");
    }
    if metadata.tokenizer.sha256 != sha256(tokenizer_path)? {
        bail!(
            "fixture tokenizer hash does not match {}",
            tokenizer_path.display()
        );
    }
    if metadata.corpus.sha256 != sha256(corpus_path)?
        || metadata.corpus.records != records.len()
        || metadata.corpus.utf8_bytes
            != records
                .iter()
                .map(|record| record.text.len())
                .sum::<usize>()
    {
        bail!(
            "fixture corpus metadata does not match {}",
            corpus_path.display()
        );
    }
    Ok(())
}

fn validate_encoding_shape(index: usize, encoding: &OracleEncoding) -> Result<()> {
    let length = encoding.ids.len();
    if encoding.type_ids.len() != length
        || encoding.offsets.len() != length
        || encoding.sequence_ids.len() != length
        || encoding.special_tokens_mask.len() != length
    {
        bail!("fixture encoding arrays differ in length at record {index}");
    }
    Ok(())
}

fn digest_ids(ids: &[u32]) -> Aggregate {
    let mut checksum = 0xcbf29ce484222325_u64;
    for &id in ids {
        checksum ^= u64::from(id);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    Aggregate {
        tokens: ids.len(),
        checksum,
    }
}

fn checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        checksum ^= u64::from(byte);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    checksum ^ 0xff
}

fn file_metadata(path: &Path) -> Result<FileMetadata> {
    file_metadata_named(path.display().to_string(), path)
}

fn file_metadata_named(name: impl Into<String>, path: &Path) -> Result<FileMetadata> {
    Ok(FileMetadata {
        path: name.into(),
        sha256: sha256(path)?,
    })
}

fn reject_benchmark_output_overwrite(
    tokenizer: &Path,
    corpus: &Path,
    vocab: &Path,
    merges: &Path,
    fixture: &Path,
    output: &Path,
) -> Result<()> {
    for input in [tokenizer, corpus, vocab, merges, fixture] {
        if output == input {
            bail!("benchmark output path must differ from every input path");
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&[9, 1, 5]), 5);
        assert_eq!(median(&[10, 2, 8, 4]), 6);
    }

    #[test]
    fn uncached_gpt2_model_matches_serialized_model() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tokenizer = load_tokenizer(&root.join("testdata/gpt2/tokenizer.json")).unwrap();
        let model = build_uncached_model(
            &tokenizer,
            &root.join("testdata/gpt2/vocab.json"),
            &root.join("testdata/gpt2/merges.txt"),
        )
        .unwrap();
        assert_eq!(
            model
                .tokenize("ĠHello")
                .unwrap()
                .iter()
                .map(|token| token.id)
                .collect::<Vec<_>>(),
            tokenizer
                .get_model()
                .tokenize("ĠHello")
                .unwrap()
                .iter()
                .map(|token| token.id)
                .collect::<Vec<_>>()
        );
    }
}

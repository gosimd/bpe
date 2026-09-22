# Compact benchmark protocol v1

One independent result per CPU/architecture profile. Raw measurements remain
unchanged when updating presentation. Six rows: GPT-2 and
Qwen3.8-Flash-Next × Model.Tokenize, Tokenizer.AppendIDs, Tokenizer.Encode.
Qwen2.5 and BLOOM remain in the extended suite; no assets or tests are retired.

## Run

Requires Python 3.9+, Go 1.27+, Git, and the pinned Rust 1.98.1 toolchain.
Cargo/Rust must be on PATH (the script also searches the usual ~/.cargo/bin).
The first Cargo build may need network access; subsequent runs use local assets.
No Python packages are needed. On macOS/Linux, from the repository root:

```sh
python3 scripts/benchmark_compact.py --machine apple-m5 \
  --environment 'MacBook Pro M5; describe power, affinity, thermal and background conditions'
```

Use a CPU/ISA label such as `apple-m5`, `xeon-skylake-avx512`, or `core-i3-avx2`,
not an SSH alias or hostname.
The ID describes the machine, **not the selected BPE backend**. The current BPE
implementation is scalar; there is no SIMD/scalar selector. Both the report and
metadata explicitly say so. When SIMD is introduced, add verified backend
selection and same-commit measurements before filling the scalar ratio column;
never substitute the heap oracle or infer a backend from CPU capabilities.
This describes BPE, not the whole pipeline: pretok/unicode dependencies retain
their normal CPU dispatch, are version-pinned, and are not forced to scalar.

Default: 20 measured samples, three complete warmup passes before each sample
of each API, one Go execution thread, no BPE cache. Each sample runs in a fresh
process. Loading, compiling, verifying HF results and allocating sufficient
AppendIDs destination capacity are outside timing. GC remains enabled with
the recorded Go settings. A timed sample is one pass through all five corpora;
records are independent and order is fixed. Every timed API consumes output IDs
in a checksum. Language timings add five clock intervals per pass. Samples are
not request-latency percentiles.

Use `--samples N` (at least 10) to select a count **before** measuring. The script
never repeats until a statistically significant result appears. Do not run
tests/builds or other measurements concurrently. It serializes its own runs via
an advisory lock; it cannot control unrelated system load. It changes neither
system power configuration nor affinity; an external taskset/affinity policy
can be used on Linux and is recorded.

`--output DIR` selects a new result directory and refuses an existing one.
Otherwise results go to:

```text
benchmarks/results/compact-v1/<machine>/<UTC timestamp>-<commit>/
    report.md
    run.json
    samples.txt
```

Only completed results are published. Local diagnostics, incomplete runs,
patches, full command/environment traces and machine-specific instructions stay
under ignored `benchmarks/runs/compact-v1/<machine>/<run>/`. They are not part
of the committed report. A comparison also retains `baseline-samples.txt`.

Retained per-machine baselines (20 samples per row):

- [Apple M5, 2026-09-22](results/compact-v1/apple-m5/20260922T100228Z-58905369/report.md)
  — battery power.
- [Xeon Skylake / AVX-512, 2026-09-22](results/compact-v1/xeon-skylake-avx512/20260922T103351Z-d36f9207/report.md)
  — KVM guest, pinned to vCPU 1; BPE is scalar.

Treat execution conditions as part of each baseline, not as a claim of peak
machine throughput or a cross-machine speedup ranking.

The published `run.json` contains measurement evidence: source hashes, recorded
commit and dirty-source flag, toolchains, resolved dependencies, measurement
flags, CPU/OS/capabilities, execution conditions, input hashes, record lengths
and computed summaries. It excludes local paths, shell commands, cache settings
and diagnostic dumps. Full execution metadata and `source.patch` are retained
only in the ignored local run directory. A source change during sampling
rejects the run; incomplete runs cannot serve as baselines.

## Inputs and correctness gate

All five existing ~1 MiB pools (English, Russian, Arabic, Japanese, Chinese)
use their pinned UTF-8-safe 4096-byte request layout. First validate original
pool/source/range hashes with the shared Go preparation tool. Then normalize
**each record independently to NFC**, outside timing, using HF Tokenizers
0.23.2 and its locked Rust dependencies. This applies to both benchmark models.
There is no re-chunking or cross-record normalization; byte sizes may change.
Source/prepared JSONL hashes and exact ordered record lengths are retained.

Generate a separate Qwen JSON with only `normalizer:NFC` replaced by null;
reject any other source normalizer. The original pinned assets are untouched.
All other pipeline settings remain unchanged. This is **Qwen3.8 / prepared NFC**,
not full Qwen raw-text compatibility. No normalization enters the Go library.

Generate fresh three-level HF oracles for both derived configurations and the
prepared inputs. Before measurement check pre-tokenized values/ranges, per-piece
BPE IDs/offsets, final Encoding metadata and appended IDs. Every sampling process
also checks its loaded data before timing. Any mismatch fails, rather than
silently dropping a language/model. Special-token insertion is disabled; added
tokens are still recognized. Padding and truncation are off. Offsets refer to
prepared text. Full-corpus generated files stay in ignored `testdata/generated/`;
ordinary Go tests never invoke Rust or preparation.

`--prepare-only` builds/prepares oracles without recording timings. The derived
catalogue uses the benchmark-only `json` pipeline setting and separate oracle
names; the existing extended catalogue and historical reports are unchanged.

## Units and comparisons

The public table uses decimal MB/s (10^6 B/s): 1000 × bytes / nanoseconds.
Raw output and JSON retain ns/op and ns/input-byte. Model bytes are the UTF-8
bytes of HF-transformed BPE pieces, excluding added tokens; Tokenizer bytes are
the prepared input text. These denominators are different: do not subtract or
sum stage rates. Aggregate each sample as one complete multilingual pass, never
average language MB/s. B/request and allocs/request divide full-pass Go
allocation counters by source-record count, also for Model.

The table reports the median rate with a distribution-free, at-least-95%
order-statistic interval for the median. Ratios are T_baseline/T_current: >1 is
faster. The independent-sample bootstrap uses 4,000 resamples and a fixed seed;
an interval containing 1 is labelled `noise`. Per-language regressions are
exploratory, not multiple-comparison-corrected tests.

For actual same-run A/B, after the harness has been committed:

```sh
python3 scripts/benchmark_compact.py --machine apple-m5 --baseline-ref <commit>
```

It archives (does not checkout/change) the baseline into a temporary directory,
requires identical measurement harness files and dependencies, builds both
versions and alternates process order per sample. Both use current pinned input
data and pass HF validation. Extra samples are kept in `baseline-samples.txt`.
A ref predating the compact harness is rejected, not retroactively benchmarked
under an invented contract. Baseline-ref arguments are existing Git refs only.

For an explicitly historical comparison:

```sh
python3 scripts/benchmark_compact.py --machine apple-m5 --previous <old-result-directory>
```

It rejects mismatched workloads, machine, toolchain, dependencies or harness.
Historical ratios are labelled: confidence intervals do **not** remove changes
in thermal conditions, power or background load. First runs leave both ratio
columns empty. No HF/tiktoken/sugarme speedup is inferred from this report;
external-library comparisons remain a separate suite.

Change protocol version when altering workloads, timed work or byte denominators.
Display-unit and artifact-retention changes do not change measured samples.
A compiler/dependency change needs a new controlled baseline,
not attribution to a BPE SIMD kernel.

## Harness tests

```sh
python3 -m unittest discover -s scripts -p 'test_benchmark_compact.py'
cargo test --locked --manifest-path tools/hf-reference/Cargo.toml --example prepare_nfc
go test ./...
```

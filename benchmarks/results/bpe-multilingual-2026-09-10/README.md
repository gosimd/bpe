# Multilingual BPE baseline — Apple M5, 2026-09-10

This is a baseline of the current scalar implementation, not an optimization
experiment. All 20 model/corpus combinations passed the differential gate.
No production merge algorithm, map representation or public API was changed.

- [Timings and allocations](timings.md): Go Model.Tokenize, HF uncached BPE,
  and the separate Go Model.AppendIDs path; ten samples per case.
- [Workload characteristics](workload.md): model sizes, ID/rank ranges, piece
  lengths, merges, actual merge-map queries and their frequency distributions.
- [Machine-readable summary](summary.json), [run metadata](run.json), raw
  Go samples in `go/`, full diagnostic aggregates in `workload/`.
- [HF results](../../../tools/hf-reference/results/bpe-multilingual-2026-09-10/):
  pinned HF 0.23.2, including the tool's other three operations. Only
  `bpe_only_uncached` is used in this comparison.

## Main observations

The English corpus alone is not representative of the broader workload.
The following are ratios of measured medians, not universal speed guarantees
or statistical confidence intervals. Every language remains visible in the
tables, including near-equality, slower cases and outliers.

- On English, HF/Go Model.Tokenize time is 1.12–1.62, depending on the model.
  On Japanese and Chinese, GPT-2 is approximately tied; Qwen2.5 takes 3–5%
  longer in Go, Qwen3.8 takes 12–13% longer, and BLOOM takes 57–72% longer.
  BLOOM also takes 20–22% longer on Arabic/Russian. This is not a general win
  over Hugging Face.
- English pieces have p99 = 13 initial symbols in every model. Japanese p99
  is 102–138, Chinese p99 is 75–96; the largest observed piece has 381 initial
  symbols (BLOOM/Japanese). These are actual BPE symbols, not transformed
  string byte lengths.
- Only 13.20–26.25% of Japanese/Chinese pieces use the heap, but those pieces
  produce 62.22–86.16% of all merge-map queries. These are query counts, not
  CPU-time shares. The next profiling target should include this long-piece
  path; extending the short-array optimization alone may miss much of the
  work. A CPU profile is still needed before choosing a new kernel.
- Merge-rule hit rate ranges from 29.67% to 93.07%; English is 89.78–93.07%.
  The hottest 1% of distinct queried pairs account for 45.25–77.26% of queries.
  This is repetition in an uncached algorithm, not a measured cache hit rate.
- The share of queries whose left ID has at least 64 outgoing rules is
  77.85–88.49% on English, versus 1.29–23.78% on Japanese/Chinese. A layout
  decision based only on English group sizes would be misleading.
- Only GPT-2's model IDs/merge operands fit uint16. The other three real
  vocabularies require wider IDs. All current merge ranks fit the existing
  packed short-array priority; the diagnostic does not impose this as a
  future model-size limit. No synthetic model was added.
- Model.AppendIDs is not invariably faster than Model.Tokenize: it still
  uses the heap for short pieces, whereas Tokenize uses the accepted array
  path. Its low allocations do not imply lower total CPU time. The measured
  2–5 allocations/pass are whole-operation totals. The destination starts
  at capacity 16 and may grow; this is not a fully preallocated
  zero-allocation-kernel measurement. Allocation stacks were not profiled.

## Scope and conditions

Apple M5, 32 GiB RAM, macOS 26.6.2 (25G83), darwin/arm64. Go 1.27.0,
GOEXPERIMENT empty, scalar backend; Rust 1.98.1 and tokenizers exactly 0.23.2.
AC power, battery 100%, low-power mode off. The desktop remained in use;
background load, affinity, P/E-core placement and frequency were not isolated.
No other benchmark or diagnostic process was deliberately run concurrently.
These are desktop baseline measurements, not dedicated-host performance bounds.

One model call processes one independently prepared HF piece. Each timing
sample traverses the entire selected corpus in fixed order; one operation is
one full-corpus pass, not one piece. Both inputs and models are already loaded.
No normalization/pre-tokenization/loading is inside BPE timing. The byte-rate
denominator is the UTF-8 length of HF-transformed pieces, NOT source text.
Models use their own HF pieces; Qwen/BLOOM full Go pipelines are not claimed.
Added-token pieces are excluded from BPE (zero exclusions in this corpus).

The five approximately 1 MiB text pools use their pinned 4 KiB UTF-8-safe
request layout: English has 256 records, the other languages 257 each.
English is WikiText-2 **train**, not the retired validation baseline.
Tokenizer revisions/licenses/hashes remain in their asset manifests. Exact
suite, corpus/oracle, model asset and source hashes are recorded in run.json
and each HF/diagnostic result. Records are independent, add_special_tokens=false,
padding/truncation off, offsets in bytes, dropout absent and BPE cache off.

Go: GOMAXPROCS=1, -cpu=1, -benchtime=1x, -count=10. The existing untimed
complete correctness pass exercises the APIs before timing; there is no
separate fixed-count Go warm-up. HF: one thread, three warm-up passes per
operation followed by ten measured passes; BPE cache_capacity(0). Cases run
sequentially, alternating HF→Go and Go→HF by case, not interleaving individual
samples. Different warm-up procedures and uncontrolled scheduling are limits
of this baseline, not hidden controls. Go map seeds are runtime-selected.

Materializing APIs have matching IDs/strings/offsets, but allocation ownership
differs: Go token values reference immutable vocabulary strings; Rust tokens
own their strings. Go includes allocation/GC costs during the timed loop;
Rust includes output destruction. The result measures these public APIs,
not isolated rank-selection kernels or equal allocator behavior. ID-only
AppendIDs has no direct HF comparison. Both harnesses keep results observable;
full ID/offset correctness checks happen outside the timer.

## Reproduction and instrumentation

Use the pinned Rust toolchain in tools/hf-reference/rust-toolchain.toml, build
with `cargo build --locked --release` from that directory, then from the root:

```sh
make test-reference
make analyze-model-workload WORKLOAD_DIR=benchmarks/results/new-run/workload
python3 scripts/benchmark_model_suite.py \
  --output benchmarks/results/new-run \
  --hf-output tools/hf-reference/results/new-run \
  --environment 'Actual CPU, RAM, OS, power, affinity and background load'
```

Use new directories; the harness rejects existing run results. See run.json
for exact executed argument vectors and model/corpus selectors. Isolated Go
cache paths and a locally selected Cargo/RUSTC executable were environment
workarounds, not compiler flags. The source checkout was dirty; the base Git
revision alone is insufficient to identify it. Source hashes identify the
measured implementation. `runner.py` preserves the exact harness bytes at
launch, matching the script hash in run.json. The current orchestration script
subsequently gained diagnostic-summary rendering and automatic harness
snapshotting; its measurement commands/loops were not changed.

The workload analyzer uses generated **test-only** copies of Tokenize and the
merge methods, replacing each merge-map read with a counted read. The source
generator checks lookup-site counts; `make analyze-model-workload` rejects a
stale generated copy. No counters or tracing branches enter production or
timed benchmarks. For each piece, traced output is checked against the normal
implementation and HF IDs/local byte offsets. Across all cases this covered
2,199,928 pieces, 14,810,658 applied merges and 44,888,439 merge-map reads.

Lookup sites: `mergeAll.1` = direct two-symbol lookup;
`updateArrayPair.1` = initial/updated short-array candidate;
`mergeHeap.1` = initial/updated heap candidate;
`mergeHeap.2` = popped candidate revalidation. A rule found by revalidation is
a hit even when its rank/result no longer matches the queued candidate.
Queries with no right neighbour are not map reads and are not counted.
Histogram quantiles use nearest rank and count every occurrence, not unique
piece strings. Full-corpus oracle JSONL remains ignored; only aggregate
statistics and measurement evidence are retained here.

## Verification

The full-corpus differential gate passed for all 20 combinations. The counted
replay separately passed all 20 cases against both production and HF. The
summary checks that Go, HF and diagnostic piece/token/input-byte counts agree;
diagnostic and HF oracle/corpus hashes were also checked for every case.
The preserved runner SHA-256 matches its recorded launch-time source hash.

`go test ./...`, `go test -race ./...`, `go vet ./...`, the instrumentation
freshness check and `go test -run=^$ -bench=. -benchtime=1x -benchmem ./...`
all passed. The last command is only a smoke check of ordinary benchmarks;
the retained timings above come from the explicitly selected corpus runs.

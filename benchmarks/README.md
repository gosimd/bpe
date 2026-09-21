# Representative Go benchmarks

## Named model/corpus suite

`benchmarks/suite.json` defines two ordered lists. `BenchmarkBPE` measures their
cross-product as `BenchmarkBPE/<model>/<corpus>/<operation>`. The default models
are GPT-2, Qwen2.5, Qwen3.8-Flash-Next and BLOOM; the corpora are five
vendored [language pools](../testdata/corpora/text/README.md): English,
Russian, Arabic, Japanese and Chinese, each approximately 1 MiB. Default pool
requests are approximately 4 KiB, with pinned UTF-8-safe request boundaries.

```sh
make benchmark-suite
make benchmark-suite MODELS=gpt2 CORPORA=ru-4k,ja-4k COUNT=10 BENCHTIME=1x
make benchmark-suite MODELS=qwen2-5,bloom COUNT=10 BENCHTIME=1x
make benchmark-suite MODELS=qwen3-8 COUNT=10 BENCHTIME=1x
make benchmark-hf-suite MODELS=gpt2 CORPORA=ru-4k,ja-4k \
  HF_ENVIRONMENT='CPU/model; OS version; power mode; background load and affinity'
```

These commands explicitly build Rust with `--locked`, prepare the selected
corpora, and generate one three-level HF 0.23.2 oracle per model/corpus pair.
`make prepare-corpora` performs only corpus conversion, without Rust. Generated
corpus JSONL, oracles and default HF timings remain under ignored
`testdata/generated/`. Neither ordinary Go tests nor the benchmark itself
builds Rust or generates missing files. Missing or stale selected data is an
error, not a silent skip; an unset `BPE_BENCH_SUITE` skips the entire opt-in suite.

To reuse prepared inputs and select individual operations:

```sh
BPE_BENCH_SUITE="$PWD/benchmarks/suite.json" BPE_HF_REPOSITORY="$PWD" \
BPE_BENCH_MODELS=gpt2 BPE_BENCH_CORPORA=ru-4k,ja-4k GOMAXPROCS=1 \
go test -run '^$' -bench '^BenchmarkBPE/[^/]+/[^/]+/ModelTokenize$' \
  -benchmem -benchtime=1x -count=10 -cpu=1 .
```

Go uses its normal benchmark calibration (explicit `1x` disables calibration);
the untimed correctness pass exercises each method once before timing, but is
not a claim that the CPU/cache is in a specified warm state. No BPE cache is
enabled in Go. HF defaults to three warm-up passes and ten measured passes,
one thread, with uncached BPE/full encode and warm-cache full encode separate.
Override with `HF_WARMUP`, `HF_ITERATIONS`, and `HF_RESULT_DIR`.
HF output paths are `<HF_RESULT_DIR>/<model>/<corpus>.json`.
`HF_ENVIRONMENT` is required for HF timings: replace the example placeholders
with the actual machine and execution conditions. Unknown power/affinity/load
conditions should be identified explicitly, not described as controlled.

### Adding a model or corpus

Add a uniquely named model with repository-relative `tokenizer`, `vocab`, and
`merges` paths and `pipeline: "model-only"`. Pin and document its source revision,
SHA-256 and license beside those assets. The benchmark derives supported BPE
options from `tokenizer.json` (unknown token, prefixes/suffixes, fuse-unknown,
byte fallback, ignored merges); it rejects non-BPE and nonzero dropout.
Exported vocabulary and ordered merge files must match `tokenizer.json` exactly,
including entries unused by the selected corpus; the loader checks this.
IDs remain uint32; the infrastructure imposes no 16-bit vocabulary limit.

`model-only` runs **ModelTokenize and ModelAppendIDs** on actual transformed HF
pieces, excluding added-token pieces. It does not claim that our Go Tokenizer
supports that model's normalizer, pre-tokenizer or post-processor. The `gpt2`
preset additionally enables PreTokenizer, TokenizerEncode and TokenizerAppendIDs
and validates the expected pipeline configuration. Other full Go pipelines
need an explicit implementation and correctness gate, not a new label. This
test tooling does not enable new pipeline presets automatically. The GPT-2
Tokenizer is constructed through the public `OpenTokenizer` JSON loader outside
timing; model-only cases retain their independent model construction.

Full Qwen tokenization is not a project goal. These models provide large
merge lists and interesting pre-tokenization configurations. NFC and all
other runtime normalization are out of scope. For dedicated pre-tokenization
comparisons we will prepare NFC-normalized Qwen text outside both libraries'
timed regions, then compare the same prepared input and prepared-text byte
offsets. Record source/prepared hashes, preparation settings, HF version and
record boundaries. Do not introduce normalization into the Go pipeline or
claim original-text offsets after external normalization. Existing model-only
cases still use HF-prepared pieces; existing HF full-pipeline measurements
keep their original scope and metadata. The prepared-text comparison is not
yet implemented by this policy change.

| Suite model | Pinned tokenizer source | BPE vocab / merges | Go operations |
| --- | --- | ---: | --- |
| gpt2 | openai-community/gpt2 | 50,257 / 50,000 | All five |
| qwen2-5 | [Qwen/Qwen2.5-0.5B](../testdata/qwen2.5/MANIFEST.md) | 151,643 / 151,387 | Model only |
| qwen3-8 | [Qwen/Qwen3.8-Flash-Next](../testdata/qwen3.8/MANIFEST.md) | 248,044 / 247,587 | Model only |
| bloom | [bigscience/bloom-560m](../testdata/bloom/MANIFEST.md) | 250,680 / 250,434 | Model only |

Model manifests pin exact revisions, upstream licenses, file sizes and
SHA-256 checksums. BLOOM's vocab/merges are deterministic exports of its
upstream tokenizer JSON using `hf-reference export-model`; other assets are
downloaded unchanged. There is no synthetic 400k benchmark corpus/model. The
tiny synthetic fixture under internal testdata is only a loader regression test.

For a new JSONL corpus, add `{"name":"my-text","path":"testdata/my-text.jsonl"}`,
one `{"text":"..."}` object per record, preserving empty records and order.
For another imported request size, add for example
`{"name":"ru-64k","pool":"ru-wikipedia","request_bytes":65536}`.
The vendored sizes are 256, 4096, 65536 and 1048576 bytes. Choose a distinct name
for every request layout. `SUITE=path/to/catalogue.json` selects another catalogue;
all asset paths remain relative to the repository root. `MODELS` and `CORPORA`
are comma-separated exact names; unknown names fail explicitly.

Every selected case validates the entire prepared corpus before timing:
Model IDs and local offsets, Model.AppendIDs IDs, and (where supported) all
pre-tokenized values/ranges and final Tokenizer IDs/offsets and AppendIDs IDs.
Loading, decompression, JSON parsing, compilation and verification are outside
timers. Each operation is one sequential full-corpus pass, not parallel batch
tokenization. The meaning of bytes, allocations and API pairs below is unchanged.

PreTokenizer now executes `gosimd/pretok v0.2.0` Pipeline and materializes owned
piece strings and source ranges, copying callback-borrowed text before retaining
it. Compilation remains outside timing. TokenizerEncode uses aligned `Walk`;
TokenizerAppendIDs uses synchronous `WalkTextUnsafe` callbacks. Record the
resolved dependency version (`GOWORK=off go list -m all`) beside new results;
historical results measured the former BPE-local scanner and are not relabeled.

### Retaining results

For an explicitly retained BPE-only comparison on all prepared cases:

```sh
make generate-suite-fixtures
make analyze-model-workload WORKLOAD_DIR=benchmarks/results/my-run/workload
python3 scripts/benchmark_model_suite.py \
  --output benchmarks/results/my-run \
  --hf-output tools/hf-reference/results/my-run \
  --environment 'Actual CPU, RAM, OS, power mode, affinity and background load'
```

The orchestration script requires a prebuilt pinned Rust executable and local
oracles. Python only orchestrates processes and summarizes output; it is not in
either timed loop. Go runs ten one-pass samples of ModelTokenize and
ModelAppendIDs (`GOMAXPROCS=1`, `-cpu=1`), with its existing untimed correctness
pass and no separate fixed warm-up. HF runs three warm-up and ten measured
passes with one thread. Its existing tool also records the other three HF
operations, but this report compares only uncached BPE with ModelTokenize.
Cases run sequentially; Go/HF order alternates by case. Raw samples, exact
commands, source/input hashes and the executing harness are retained.

### Untimed model workload analysis

`make analyze-model-workload` uses already generated local oracles. It does not
build Rust or measure performance. `MODELS`, `CORPORA`, `SUITE` and
`WORKLOAD_DIR` select the cases and output directory, defaulting to the full
suite and ignored `testdata/generated/model-workload`.

The analyzer derives a **test-only** copy of the current production merge
methods with counted map lookups; no observer, counter, branch or build tag is
added to the library. The command checks that the generated copy is current.
After intentional merge-code changes regenerate it with
`go run ./tools/workload-instrument`. Each analyzed piece must agree with the
unmodified implementation and HF IDs/local byte offsets before output is saved.

JSON records vocabulary/rule counts, ID/rank ranges, initial symbol and
transformed UTF-8 byte lengths, output lengths, merge counts, no-merge frequency,
dispatch paths, lookup hits/misses, per-site counts, pair query frequency
histograms/top pairs and query-weighted rule-group sizes by left ID. A hit
means a rule exists, not a cache hit or an applied merge. Heap revalidation
lookups are included. Applied merges equal initial minus final symbols;
ignore-merges vocabulary shortcuts are counted separately. Added-token pieces
are excluded. These are Model.Tokenize diagnostics, not AppendIDs or HF lookup
traces. Full corpus oracles stay ignored; compact aggregate reports may be
retained with baseline results.

### Result metadata

HF JSON results include hashes, corpus counts, modes, warm-up, iteration and
thread counts and environment data. For a retained comparison, place those
results in `tools/hf-reference/results/` and Go output in `benchmarks/results/`.
Keep a copy of the exact selected catalogue and the vendored corpus manifest
(or their hashes), model asset hashes, Go/Rust versions, CPU, OS, architecture,
exact commands and execution conditions alongside the results. Report each
language/model separately; do not average away regressions. A smoke run is not
a retained performance baseline.

## Operations and retired baseline

`make benchmark-go-reference` is an alias for the named `benchmark-suite`.
`make generate-fixtures` generates the selected suite's oracles, and
`make test-reference` checks those cases at the model and supported pipeline
boundaries, including complete Encoding metadata and AppendIDs record bounds.
Ordinary `go test` does not generate data or invoke Rust.

The old `BenchmarkGPT2WikiText2` and its validation corpus have been removed.
Retained reports describe that historical workload, not the current English
WikiText-2 train pool. Their old commands require the corresponding Git revision.

The benchmark loads the corpus, oracle, and compiled model before starting its
timers. Before any measurement it checks all produced IDs and the
pre-tokenizer's transformed values and offsets, model offsets and final offsets
against the oracle. The full differential test remains responsible for the other
Encoding metadata. Inputs remain
in memory and records and pieces retain their original order.

The five cases are:

- `PreTokenizer`: AddedToken matching plus GPT-2 regex splitting and ByteLevel
  mapping, including output construction;
- `ModelTokenize`: public token/ID/offset output for each prepared BPE piece;
- `ModelAppendIDs`: ID-only model path with a caller-owned destination reused
  between pieces;
- `TokenizerEncode`: public raw-text Encoding with special tokens disabled;
- `TokenizerAppendIDs`: ID-only raw-text path with a destination reused between
  records.

Only two cross-language pairs have equivalent outputs:

- HF `Model::tokenize` with cache disabled ↔ Go `Model.Tokenize`;
- HF `Tokenizer::encode` with cache disabled ↔ Go `Tokenizer.Encode`.

The pre-tokenizer rows are useful pipeline diagnostics, although their
materialized internal result types differ. The AppendIDs cases intentionally
omit token strings, offsets, and Encoding metadata; report them as Go inference
hot paths rather than direct speedups over the materializing HF APIs.

Each operation covers a whole corpus pass. Consequently, `B/op` and
`allocs/op` are totals per corpus, not per record or token. `Model*` throughput
uses transformed-piece bytes; the other cases use source-text bytes.

## Internal merge shapes

`BenchmarkMergeShapes` in `internal/model` measures the production merge
dispatcher on prepared, reusable symbols without model loading or output
construction. It covers one/two-symbol pieces, the short-array boundary,
longer heap inputs, merge-dense `ab` and no-merge `x` strings. These synthetic
kernel timings complement, but do not replace, the public corpus benchmark.

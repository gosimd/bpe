# Retained Go benchmark results

Keep reviewed, machine-specific output from the representative Go benchmarks
in this directory. Every retained result needs a companion report recording:

- the corpus, tokenizer, and oracle hashes;
- Go version, OS, architecture, CPU, power and load conditions;
- thread count, `-benchtime`, `-count`, and exact command;
- source hashes or a clean Git revision;
- medians and ranges computed from the unmodified raw output;
- the exact HF result used for a cross-language comparison.

Do not compare Go and HF cases with different output contracts. Do not present
the ID-only AppendIDs paths as direct HF API speedups. Retain regressions and
raw samples, not only summarized wins.

## Current multilingual baseline

[BPE baseline and workload, 2026-09-10](bpe-multilingual-2026-09-10/README.md)
covers GPT-2, Qwen2.5, Qwen3.8 and BLOOM on all five language pools. It keeps
per-case raw Go/HF timings and untimed merge-lookup statistics, including
regressions; it does not claim full Go pipelines for the model-only entries.

## Historical workload notice

The retained `gpt2-wikitext2-*` reports measure the retired 3,760-record
WikiText-2 validation corpus. That input directory and benchmark were removed;
use the report's source revision to reproduce its commands. The current English
corpus uses WikiText-2 train with different request boundaries. These are not
interchangeable workloads. Historical timings and metadata are left unchanged.

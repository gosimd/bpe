# Compact BPE baseline — xeon-skylake-avx512

UTC: 2026-09-22T10:33:51.114801+00:00 · commit: `d36f9207bda5ee04978160fccb4df75324957153` · protocol: `compact-v1`

CPU: Intel Xeon Processor (Skylake, IBRS, no TSX) · Linux 7.0.0-15-generic / x86_64
go version go1.27.0 linux/amd64 · BPE backend: **scalar** (no SIMD dispatch implemented).
pretok/unicode use their default dependency dispatch; this is not a forced-scalar full pipeline.
Single thread; 3 warmup passes before each sample; 20 samples per row.
Five NFC-prepared language pools (en/ru/ar/ja/zh), independently encoded ~4 KiB requests.
Normalization, loading, compilation, allocation of dst, and HF correctness checks are outside timing.
No BPE cache. Special-token insertion off; AddedToken matching on. AppendIDs reuses sufficient dst capacity.

| Model | API | MB/s ↑ (95% median CI) | B/request ↓ | allocs/request ↓ | × previous | × scalar |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | Model.Tokenize | 6.06 [5.92, 6.24] | 191843 | 906.92 | — | — |
| GPT-2 | Tokenizer.AppendIDs | 3.66 [3.56, 3.71] | 0 | 0.00 | — | — |
| GPT-2 | Tokenizer.Encode | 1.99 [1.91, 2.03] | 716453 | 477.48 | — | — |
| Qwen3.8 / prepared NFC | Model.Tokenize | 4.84 [4.51, 5.08] | 138513 | 865.16 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.AppendIDs | 2.6 [2.53, 2.67] | 0 | 0.00 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.Encode | 1.93 [1.9, 1.96] | 324768 | 454.60 | — | — |

1 MB = 10⁶ bytes; MB/s = 1000 × input bytes / nanoseconds. Model counts transformed-piece UTF-8 bytes; Tokenizer counts prepared source bytes.
One timed op is a complete five-language pass; allocations are divided by source-record count, including for Model.
Timed paths consume every output ID in a checksum. Language timing boundaries add five clock intervals per pass.
CI: distribution-free order-statistic interval for the median. Ratios, when present, use an independent-sample bootstrap (4,000 resamples, seed 1729).
Scalar comparison is unavailable: both current production merge paths are scalar; no heap-oracle comparison is substituted.

## Execution conditions

Xeon Skylake KVM guest, 2 vCPUs; AVX2/AVX-512 exposed; process pinned to guest CPU 1; physical host placement, frequency and background load uncontrolled.

Affinity: [1]. Power source: unknown.

Raw samples: [samples.txt](samples.txt). Measurement metadata: [run.json](run.json).
HF 0.23.2 is a correctness oracle here, not a timed competitor.

First baseline: no previous version was supplied; no speedup is claimed.

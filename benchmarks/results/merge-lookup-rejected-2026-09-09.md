# Rejected experiment: SIMD merge-rule lookup

Decision (2026-09-09): retain Go map. The experimental implementation, build
tags, runner and raw artifacts were removed at the maintainer's request.
This is a historical summary, not a reproducible benchmark bundle.

## Hypothesis and setup

Replace `map[pair]merge` with an immutable hash-bucket table whose contiguous
keys are separate from rank/newID values. Compare the same table with scalar
and ARM64 NEON matchers. Select uint32 packed pairs when vocabulary IDs fit
uint16, otherwise uint64 pairs; compare four or two keys per vector.
The BPE merge algorithm and materialized output remain unchanged.

Environment: Apple M5, macOS 26.6.2 build 25G83 arm64, Go 1.27.0,
`GOEXPERIMENT=simd` for all variants, `GOMAXPROCS=1`, no affinity or thermal
control, ordinary desktop load, AC reported with battery discharging.
Workload: pinned GPT-2 / WikiText-2 raw validation, 3,760 records,
219,615 prepared BPE pieces. No result cache. Model construction and HF
0.23.2 oracle validation occurred outside timing.

Six rounds used all six variant order permutations. Each corpus sample
averaged five passes, with untimed validation and Go calibration warming the
paths. The measured test-binary arguments were:

```sh
-test.run='^$' -test.bench='^BenchmarkGPT2WikiText2$/^ModelTokenize$' \
  -test.benchtime=5x -test.count=1 -test.cpu=1 -test.benchmem
```

## Observations

| Merge-rule lookup | Median Model.Tokenize per corpus | Min–max | Change vs map |
| --- | ---: | ---: | ---: |
| Go map | 57.202 ms | 56.208–60.166 ms | — |
| Scalar bucket table | 56.750 ms | 55.419–59.061 ms | −0.79% |
| NEON bucket table | 59.320 ms | 58.676–60.174 ms | +3.70% |

The scalar table won only two of six paired rounds; its small median
advantage was not repeatable. NEON lost five of six rounds against map.
These small samples do not establish universal performance differences.
Timed allocations were effectively unchanged: about 39.83 MB and 439,232
allocations per corpus pass.

Isolated mixed hit/miss lookups with uint32 keys favored NEON: 5.808 versus
8.102 ns/lookup for map (about 28% lower latency), but that synthetic-query
win did not transfer to Model.Tokenize. The GPT-2 experimental table added
1,792,016 bytes of backing storage because the source map was also retained.

Assembly confirmed `CMEQ.4S` and `CMEQ.2D`. The NEON matcher remained a
separate call and constructed/reduced a match mask; scalar scanning was
inlined. These are plausible costs, not independently measured causes.
Map-parity, HF differential, race, fuzz and fallback-build checks passed.

## Conclusion

Reject this implementation, not every possible SIMD hash-table design.
Do not promote a lookup microbenchmark win without a repeatable public
Model.Tokenize improvement. Any revisit should reconsider call/mask overhead,
bucket metadata and padding, memory cost, and real merge-query distributions.
The accepted short-piece merge optimization is independent and remains intact.

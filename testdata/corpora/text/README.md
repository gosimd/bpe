# Multilingual corpus sources

These are self-contained benchmark inputs. `MANIFEST.json` identifies the
compressed texts, extracted text hashes, source metadata and request ranges.
Each `*.source.json` records the upstream dataset, pinned revision, shard,
selected rows, transformations and attribution. No network or sibling checkout
is required. The manifest contains only the five text pools used here.

| Pool | Language | UTF-8 bytes | Requests at 4,096 bytes |
| --- | --- | ---: | ---: |
| en-wikitext2 | English | 1,048,576 | 256 |
| ru-wikipedia | Russian | 1,048,575 | 257 |
| ar-wikipedia | Arabic | 1,048,576 | 257 |
| ja-wikipedia | Japanese | 1,048,575 | 257 |
| zh-wikipedia | Chinese | 1,048,574 | 257 |

Each pool includes the original gzip, source metadata (dataset revisions,
article titles/URLs, selection and transformations), and exact request ranges
for 256, 4,096, 65,536 and 1,048,576 bytes. Ranges are contiguous and round down
to UTF-8 boundaries; the final request may be shorter. Each request is encoded
independently. Changing the request size changes tokenization at boundaries.

`internal/benchsuite` checks compressed/uncompressed text hashes, source metadata
hashes, range hashes, counts and UTF-8 boundaries before converting a pool to
deterministic JSONL under ignored `testdata/generated/corpora/`. The conversion
does not normalize, trim, join or re-split the requests. `make prepare-corpora`
performs this conversion without Rust.

The English pool comes from WikiText-2 **train**: nonempty records were selected,
truncated, joined and then divided into requests. The former 3,760-record
WikiText-2 **validation** workload has been retired to keep one English corpus.
They were not identical inputs. Historical benchmark reports still refer to
the validation workload and cannot be compared directly with this train pool.

See `LICENSE.md` and each `*.source.json` for attribution and licensing:
WikiText-2 is CC BY-SA 4.0; the Wikipedia pools carry the CC BY-SA 3.0/GFDL
notices recorded by the source corpus. Keep these notices with redistributed
data and derived workloads. No source text was changed during this import.

# WikiText-2 raw validation test data

This directory contains the validation split of the `wikitext-2-raw-v1`
configuration. Corpus records remain separate because joining them would
change pre-tokenization at record boundaries.

- Source: <https://huggingface.co/datasets/Salesforce/wikitext>
- Revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Configuration: `wikitext-2-raw-v1`
- Split: `validation`
- Records: 3,760, including 1,299 records whose `text` value is empty
- Retrieved: 2026-09-08
- Licenses declared by the source dataset: CC BY-SA 3.0 and GFDL

| File | Origin | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `validation.parquet` | [Pinned source](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet) | 657,209 | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| `validation.jsonl` | Derived from `validation.parquet` | 1,194,397 | `d8b897d49c70e35db334af33364549e4db97c89fcfaf5917d306208abcd7587d` |

`validation.jsonl` contains one object of the form `{"text":"..."}` per
physical line. Empty source records are represented as `{"text":""}` lines;
they are not omitted or converted to blank physical lines. Record order and
text values exactly match the Parquet source.

Regenerate it with Python 3.12 and PyArrow 21.0.0:

```sh
python3 scripts/generate_wikitext2_jsonl.py \
  testdata/wikitext2/validation.parquet \
  testdata/wikitext2/validation.jsonl \
  --expected-rows 3760
```

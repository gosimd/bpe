# Corpus licenses and attribution

These text excerpts are **not covered by the Go library's MIT license**.

- `en-wikitext2`: Salesforce WikiText (`wikitext-2-raw-v1`), CC BY-SA 4.0.
  [Dataset and attribution](https://huggingface.co/datasets/Salesforce/wikitext),
  [license](https://creativecommons.org/licenses/by-sa/4.0/).
  Dataset authors: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher;
  underlying text originates from Wikipedia contributors.
- `ru-wikipedia`, `ja-wikipedia`, `ar-wikipedia`, `zh-wikipedia`: Wikimedia
  Wikipedia contributors, from the `20231101` cleaned article snapshot.
  [Dataset license statement](https://huggingface.co/datasets/wikimedia/wikipedia#licensing-information),
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/),
  [GFDL](https://www.gnu.org/licenses/fdl-1.3.html) where applicable.
  Every selected article's title and original URL is retained in its corpus
  `.source.json`; attribution/history is available at those URLs.

Changes from the datasets: selected rows, UTF-8-safe truncation to 32 KiB per row,
joining with two LF bytes, and truncation of the resulting pool to approximately
1 MiB. Original content is otherwise unchanged. Exact source dataset revisions,
shards, row identifiers, source hashes, extracted ranges, and output ranges are
recorded in each `.source.json`. Request boundaries are benchmark metadata,
not additional edits to the text.

Keep this notice and the source manifests with redistributed corpus files.

# BPE workload: generated summary

One untimed pass per model/corpus; every traced piece is checked against production and HF IDs/local byte offsets. Full histograms, lookup-site counters and top queried pairs are in workload/.

## Model tables

Rules/left includes only left IDs having rules. IDs exclude pipeline added tokens. Max rank is zero-based.

| Model | Vocab entries | Merge rules | ID range | Max operand ID | Max rank | Rules/left p50 / p90 / p99 / max |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| gpt2 | 50,257 | 50,000 | 0–50256 | 48702 | 49999 | 2 / 9 / 70 / 330 |
| qwen2-5 | 151,643 | 151,387 | 0–151642 | 151560 | 151386 | 1 / 10 / 62 / 956 |
| qwen3-8 | 248,044 | 247,587 | 0–248043 | 247989 | 247586 | 2 / 10 / 68 / 948 |
| bloom | 250,680 | 250,434 | 0–250679 | 249351 | 250433 | 2 / 10 / 67 / 1462 |

## Piece shape

Symbols are the actual initial BPE symbols, not UTF-8 byte length. UTF-8 bytes refer to HF-transformed pieces, not original text. Quantiles are nearest-rank, weighted by occurrences. Merge count is initial minus final symbols (ignore-merges direct-vocab bypass is recorded separately as zero merges).

| Model | Corpus | Pieces | Symbols mean | Symbols p50 / p90 / p99 / max | UTF-8 bytes mean / p99 / max | Merges mean / max | No merge % |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| gpt2 | en-4k | 199,960 | 5.24 | 5 / 9 / 13 / 31 | 6.28 / 14 / 62 | 4.09 / 19 | 1.15 |
| gpt2 | ru-4k | 98,713 | 10.62 | 10 / 23 / 31 / 45 | 20.97 / 62 / 90 | 4.43 / 21 | 16.47 |
| gpt2 | ar-4k | 123,496 | 8.49 | 9 / 15 / 19 / 35 | 16.81 / 38 / 70 | 3.84 / 18 | 15.83 |
| gpt2 | ja-4k | 74,514 | 14.07 | 3 / 45 / 102 / 225 | 27.75 / 204 / 450 | 6.83 / 125 | 21.73 |
| gpt2 | zh-4k | 85,560 | 12.26 | 3 / 39 / 75 / 156 | 23.97 / 150 / 312 | 3.59 / 48 | 31.90 |
| qwen2-5 | en-4k | 214,997 | 4.88 | 4 / 9 / 13 / 31 | 5.84 / 14 / 62 | 3.76 / 23 | 10.78 |
| qwen2-5 | ru-4k | 102,674 | 10.21 | 9 / 23 / 31 / 45 | 20.17 / 62 / 90 | 8.16 / 39 | 21.90 |
| qwen2-5 | ar-4k | 123,724 | 8.48 | 9 / 15 / 19 / 35 | 16.78 / 38 / 70 | 6.66 / 27 | 12.46 |
| qwen2-5 | ja-4k | 56,746 | 18.48 | 7 / 54 / 111 / 228 | 36.44 / 222 / 456 | 13.74 / 181 | 25.99 |
| qwen2-5 | zh-4k | 62,370 | 16.81 | 9 / 45 / 81 / 156 | 32.88 / 162 / 312 | 12.61 / 118 | 24.95 |
| qwen3-8 | en-4k | 214,979 | 4.88 | 4 / 9 / 13 / 40 | 5.84 / 14 / 80 | 3.77 / 33 | 10.78 |
| qwen3-8 | ru-4k | 102,571 | 10.22 | 9 / 23 / 31 / 45 | 20.19 / 62 / 90 | 8.59 / 41 | 22.34 |
| qwen3-8 | ar-4k | 118,768 | 8.83 | 9 / 15 / 21 / 35 | 17.48 / 42 / 70 | 7.27 / 28 | 13.39 |
| qwen3-8 | ja-4k | 56,746 | 18.48 | 7 / 54 / 111 / 228 | 36.44 / 222 / 456 | 14.44 / 193 | 26.16 |
| qwen3-8 | zh-4k | 62,370 | 16.81 | 9 / 45 / 81 / 156 | 32.88 / 162 / 312 | 13.01 / 126 | 25.04 |
| bloom | en-4k | 197,470 | 5.31 | 5 / 9 / 13 / 40 | 6.36 / 14 / 80 | 4.15 / 27 | 0.92 |
| bloom | ru-4k | 90,897 | 11.54 | 11 / 23 / 33 / 68 | 22.78 / 66 / 135 | 8.50 / 51 | 10.31 |
| bloom | ar-4k | 108,922 | 9.63 | 9 / 15 / 21 / 67 | 19.06 / 42 / 114 | 8.19 / 44 | 3.92 |
| bloom | ja-4k | 44,928 | 23.34 | 5 / 72 / 138 / 381 | 46.03 / 276 / 762 | 16.38 / 271 | 6.95 |
| bloom | zh-4k | 59,523 | 17.62 | 5 / 51 / 96 / 359 | 34.45 / 188 / 704 | 13.82 / 275 | 4.57 |

## Actual merge-map queries

Hit = rule exists for the queried pair, NOT a BPE-cache hit and NOT necessarily an applied merge. Includes initial candidates, neighbour refreshes and heap revalidation. Frequencies include repeated queries, within and across pieces. Top 1% uses ceil(1% of unique queried pairs). Left>=64 is the fraction of all queries whose left ID has at least 64 merge rules, including queries that miss.

| Model | Corpus | Queries | Queries/piece | Hit % | Unique pairs | Unique hit rules | Top 1% query share % | Queries with left>=64 % | Heap pieces % | Heap queries % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | en-4k | 1,681,639 | 8.41 | 89.78 | 51,390 | 24,387 | 61.33 | 77.85 | 0.00 | 0.00 |
| gpt2 | ru-4k | 1,704,009 | 17.26 | 29.67 | 3,504 | 1,455 | 45.25 | 5.11 | 0.75 | 3.46 |
| gpt2 | ar-4k | 1,668,140 | 13.51 | 33.65 | 6,180 | 2,860 | 62.77 | 9.26 | 0.00 | 0.01 |
| gpt2 | ja-4k | 2,176,926 | 29.21 | 39.51 | 33,591 | 4,792 | 58.58 | 1.29 | 14.47 | 72.59 |
| gpt2 | zh-4k | 1,635,455 | 19.11 | 30.26 | 30,727 | 6,748 | 46.79 | 3.02 | 13.20 | 62.22 |
| qwen2-5 | en-4k | 1,670,072 | 7.77 | 91.88 | 52,436 | 25,986 | 61.75 | 84.75 | 0.00 | 0.00 |
| qwen2-5 | ru-4k | 2,285,220 | 22.26 | 50.92 | 42,366 | 4,480 | 72.24 | 4.07 | 0.73 | 3.93 |
| qwen2-5 | ar-4k | 2,094,239 | 16.93 | 53.43 | 37,814 | 5,786 | 75.26 | 13.09 | 0.00 | 0.01 |
| qwen2-5 | ja-4k | 2,943,811 | 51.88 | 46.53 | 150,504 | 10,240 | 67.31 | 8.40 | 20.04 | 76.31 |
| qwen2-5 | zh-4k | 2,878,440 | 46.15 | 45.92 | 295,767 | 19,476 | 63.33 | 19.42 | 19.95 | 70.53 |
| qwen3-8 | en-4k | 1,671,336 | 7.77 | 93.07 | 53,066 | 28,221 | 61.85 | 88.49 | 0.00 | 0.01 |
| qwen3-8 | ru-4k | 2,309,419 | 22.52 | 53.81 | 54,993 | 12,427 | 74.66 | 20.66 | 0.73 | 3.99 |
| qwen3-8 | ar-4k | 2,132,203 | 17.95 | 54.68 | 48,283 | 9,792 | 76.94 | 16.99 | 0.01 | 0.04 |
| qwen3-8 | ja-4k | 3,052,137 | 53.79 | 47.49 | 189,737 | 14,809 | 68.41 | 12.63 | 20.04 | 76.51 |
| qwen3-8 | zh-4k | 2,954,013 | 47.36 | 46.67 | 334,179 | 28,651 | 63.38 | 23.78 | 19.95 | 70.74 |
| bloom | en-4k | 1,704,418 | 8.63 | 92.03 | 53,129 | 25,108 | 61.54 | 87.72 | 0.00 | 0.01 |
| bloom | ru-4k | 2,237,738 | 24.62 | 46.20 | 27,024 | 2,566 | 68.14 | 4.82 | 1.18 | 5.67 |
| bloom | ar-4k | 2,159,373 | 19.82 | 57.99 | 58,165 | 15,928 | 77.26 | 37.25 | 0.01 | 0.07 |
| bloom | ja-4k | 2,932,925 | 65.28 | 46.15 | 135,568 | 11,685 | 66.97 | 10.57 | 26.25 | 86.16 |
| bloom | zh-4k | 2,996,926 | 50.35 | 48.17 | 344,063 | 27,102 | 62.92 | 22.41 | 21.52 | 77.16 |

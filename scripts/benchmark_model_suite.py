#!/usr/bin/env python3
"""Run sequential Go/HF baselines on prepared oracles; Python is never timed.

The pinned Rust executable must already be built with Cargo --locked. This
script neither downloads data nor invokes Rust during ordinary Go testing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hf-output", required=True, type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--hf-warmup", type=int, default=3)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if args.samples < 2 or args.hf_warmup < 0 or not args.environment.strip():
        parser.error("samples >= 2, warmup >= 0 and an environment description are required")
    if args.summarize:
        summarize(args.output, args.hf_output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "run.json"
    if metadata_path.exists():
        parser.error("output already contains a run; choose a new output directory")
    # Keep the executing harness itself: report-only edits after a long run
    # must not make its recorded source hash impossible to reproduce.
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    suite = json.loads(Path("benchmarks/suite.json").read_text())
    env = os.environ.copy()
    env.update(BPE_BENCH_SUITE=str(root / "benchmarks/suite.json"),
               BPE_HF_REPOSITORY=str(root), GOMAXPROCS="1")
    env.pop("BPE_WORKLOAD_OUTPUT", None)
    binary = root / "testdata/generated/model-baseline.test"
    binary.parent.mkdir(parents=True, exist_ok=True)
    capture = lambda command: subprocess.check_output(command, text=True).strip()
    metadata = dict(schema=1, started_unix=time.time(), complete=False,
                    environment=args.environment, suite=suite, samples=args.samples,
                    hf_warmup=args.hf_warmup, go_benchtime="1x",
                    go_warmup="untimed complete correctness pass; no separate fixed warmup",
                    order="catalogue order; alternating HF/Go and Go/HF per case; no concurrent measurements",
                    go_version=capture(["go", "version"]),
                    go_settings=capture(["go", "env", "GOEXPERIMENT", "GOOS", "GOARCH"]),
                    git_head=capture(["git", "rev-parse", "HEAD"]),
                    git_status=capture(["git", "status", "--short"]),
                    commands=[], sha256={})
    sources = list(Path(".").glob("*.go")) + list(Path("internal").rglob("*.go"))
    sources += list(Path("tools").glob("*/src/*.rs")) + list(Path("tools").glob("*/*.go"))
    sources += [Path("scripts/benchmark_model_suite.py"), Path("Makefile"), Path("go.mod"),
                Path("benchmarks/suite.json"), Path("testdata/corpora/text/MANIFEST.json"),
                Path("tools/hf-reference/Cargo.toml"), Path("tools/hf-reference/Cargo.lock"),
                Path("tools/hf-reference/rust-toolchain.toml")]
    metadata["sha256"] = {str(p): sha256(p) for p in sorted(set(sources))}

    def run(command, extra_env=None, output=None):
        metadata["commands"].append(dict(argv=list(map(str, command)), env=extra_env or {}))
        write_json(metadata_path, metadata)
        process = subprocess.run(list(map(str, command)), env=dict(env, **(extra_env or {})),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(shlex.join(list(map(str, command))) + "\n" + process.stdout)

    run(["go", "test", "-c", "-o", binary, "."], output=args.output / "build.txt")
    metadata["sha256"][str(binary.relative_to(root))] = sha256(binary)
    reference = Path("tools/hf-reference/target/release/hf-reference")
    metadata["sha256"][str(reference)] = sha256(reference)
    for i, (model, corpus) in enumerate((m, c) for m in suite["models"] for c in suite["corpora"]):
        name, language = model["name"], corpus["name"]
        oracle = Path("testdata/generated/suite") / name / (language + ".jsonl")
        # The oracle metadata is the authoritative path of the prepared input.
        with oracle.open() as stream:
            header = json.loads(stream.readline())
        corpus_path = Path(header["corpus"]["path"])
        hf_path = args.hf_output / name / (language + ".json")
        go_path = args.output / "go" / name / (language + ".txt")
        if hf_path.exists() or go_path.exists():
            raise RuntimeError("refusing to overwrite existing case results")
        for path in [oracle, corpus_path] + [Path(model[k]) for k in ("tokenizer", "vocab", "merges")]:
            metadata["sha256"][str(path)] = sha256(path)
        hf = [reference, "benchmark", "--tokenizer", model["tokenizer"],
              "--vocab", model["vocab"], "--merges", model["merges"],
              "--corpus", corpus_path, "--fixture", oracle, "--output", hf_path,
              "--threads", "1", "--warmup", str(args.hf_warmup),
              "--iterations", str(args.samples), "--environment", args.environment]
        go = [binary, "-test.run=^$", "-test.bench=^BenchmarkBPE/[^/]+/[^/]+/Model(Tokenize|AppendIDs)$",
              "-test.benchmem", "-test.benchtime=1x", "-test.count=" + str(args.samples), "-test.cpu=1"]
        commands = [(hf, {}, args.output / "hf-logs" / name / (language + ".txt")),
                    (go, dict(BPE_BENCH_MODELS=name, BPE_BENCH_CORPORA=language), go_path)]
        if i % 2:
            commands.reverse()
        for command, extra_env, output in commands:
            print(f"{name}/{language}: {'Go' if command is go else 'HF'}", flush=True)
            run(command, extra_env, output)
    metadata.update(complete=True, finished_unix=time.time())
    write_json(metadata_path, metadata)
    summarize(args.output, args.hf_output)


def summarize(output, hf_output):
    metadata = json.loads((output / "run.json").read_text())
    if not metadata["complete"]:
        raise RuntimeError("refusing to summarize an incomplete baseline")
    rows = []
    for m in metadata["suite"]["models"]:
        for c in metadata["suite"]["corpora"]:
            name, corpus = m["name"], c["name"]
            go = {}
            for line in (output / "go" / name / (corpus + ".txt")).read_text().splitlines():
                fields = line.split()
                if not fields or not fields[0].startswith("BenchmarkBPE/"):
                    continue
                operation = fields[0].split("/")[-1]
                if int(fields[1]) != 1:
                    raise RuntimeError("expected one full corpus pass per Go sample")
                go.setdefault(operation, []).append({fields[i+1]: float(fields[i]) for i in range(2, len(fields), 2)})
            if set(go) != {"ModelTokenize", "ModelAppendIDs"} or any(len(v) != metadata["samples"] for v in go.values()):
                raise RuntimeError("incomplete Go samples")
            hf = json.loads((hf_output / name / (corpus + ".json")).read_text())
            bpe = next(x for x in hf["measurements"] if x["operation"] == "bpe_only_uncached")
            if bpe["cache_capacity"] != 0 or len(bpe["iteration_nanoseconds"]) != metadata["samples"]:
                raise RuntimeError("unexpected HF BPE configuration")
            for sample in go["ModelTokenize"] + go["ModelAppendIDs"]:
                if (sample["tokens/op"] != bpe["tokens_per_iteration"] or
                    sample["pieces/op"] != bpe["pieces_per_iteration"] or
                    sample["input-bytes/op"] != bpe["operation_input_bytes_per_iteration"]):
                    raise RuntimeError("Go/HF workload mismatch")
            row = dict(model=name, corpus=corpus, hf_bpe=bpe, go={})
            for operation, samples in go.items():
                row["go"][operation] = {key: dict(median=statistics.median(s[key] for s in samples),
                                                  min=min(s[key] for s in samples), max=max(s[key] for s in samples))
                                         for key in samples[0]}
            row["hf_over_go_ratio"] = statistics.median(bpe["iteration_nanoseconds"]) / row["go"]["ModelTokenize"]["ns/op"]["median"]
            rows.append(row)
    write_json(output / "summary.json", rows)
    lines = ["# BPE timings: generated summary", "", "Median and [min, max], milliseconds per full corpus pass. Ratios are descriptive, not confidence intervals.", "",
             "| Model | Corpus | Go Tokenize ms | HF uncached BPE ms | HF / Go | Go AppendIDs ms | Tokenize B/op | Tokenize allocs/op | AppendIDs B/op | AppendIDs allocs/op |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        tok, ids = row["go"]["ModelTokenize"], row["go"]["ModelAppendIDs"]
        fmt = lambda d: f'{d["median"]/1e6:.2f} [{d["min"]/1e6:.2f}, {d["max"]/1e6:.2f}]'
        samples = row["hf_bpe"]["iteration_nanoseconds"]
        hf = dict(median=statistics.median(samples), min=min(samples), max=max(samples))
        lines.append(f'| {row["model"]} | {row["corpus"]} | {fmt(tok["ns/op"])} | {fmt(hf)} | {row["hf_over_go_ratio"]:.2f} | {fmt(ids["ns/op"])} | {tok["B/op"]["median"]:.0f} | {tok["allocs/op"]["median"]:.0f} | {ids["B/op"]["median"]:.0f} | {ids["allocs/op"]["median"]:.0f} |')
    lines += ["", "Throughput (transformed-piece UTF-8 bytes), ns/input-byte, tokens/s, all raw samples and other metrics are in summary.json and go/. AppendIDs is not a direct HF API comparison."]
    (output / "timings.md").write_text("\n".join(lines) + "\n")
    summarize_workload(output, rows)


def summarize_workload(output, rows):
    workloads = []
    for row in rows:
        path = output / "workload" / row["model"] / (row["corpus"] + ".json")
        value = json.loads(path.read_text())
        s, hf = value["stats"], row["hf_bpe"]
        if (s["pieces"] != hf["pieces_per_iteration"] or
                s["final_tokens"]["sum"] != hf["tokens_per_iteration"] or
                s["piece_utf8_bytes"]["sum"] != hf["operation_input_bytes_per_iteration"] or
                s["queries_per_piece"]["sum"] != s["lookup"]["queries"] or
                s["pair_query_frequencies"]["sum"] != s["lookup"]["queries"]):
            raise RuntimeError("diagnostic/timing workload counts differ")
        workloads.append(value)
    lines = ["# BPE workload: generated summary", "",
             "One untimed pass per model/corpus; every traced piece is checked against production and HF IDs/local byte offsets. Full histograms, lookup-site counters and top queried pairs are in workload/.", "",
             "## Model tables", "",
             "Rules/left includes only left IDs having rules. IDs exclude pipeline added tokens. Max rank is zero-based.", "",
             "| Model | Vocab entries | Merge rules | ID range | Max operand ID | Max rank | Rules/left p50 / p90 / p99 / max |",
             "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    seen = set()
    quantiles = lambda d: f'{d["p50"]} / {d["p90"]} / {d["p99"]} / {d["max"]}'
    for value in workloads:
        name, s = value["model"], value["stats"]
        if name in seen:
            continue
        seen.add(name)
        lines.append(f'| {name} | {s["vocab_entries"]:,} | {s["merge_rules"]:,} | {s["min_id"]}–{s["max_id"]} | {s["merge_operand_max_id"]} | {s["max_rank"]} | {quantiles(s["rules_per_left"])} |')
    lines += ["", "## Piece shape", "",
              "Symbols are the actual initial BPE symbols, not UTF-8 byte length. UTF-8 bytes refer to HF-transformed pieces, not original text. Quantiles are nearest-rank, weighted by occurrences. Merge count is initial minus final symbols (ignore-merges direct-vocab bypass is recorded separately as zero merges).", "",
              "| Model | Corpus | Pieces | Symbols mean | Symbols p50 / p90 / p99 / max | UTF-8 bytes mean / p99 / max | Merges mean / max | No merge % |",
              "| --- | --- | ---: | ---: | --- | --- | --- | ---: |"]
    for value in workloads:
        s = value["stats"]
        d, b, m = s["initial_symbols"], s["piece_utf8_bytes"], s["merges_per_piece"]
        lines.append(f'| {value["model"]} | {value["corpus"]} | {s["pieces"]:,} | {d["mean"]:.2f} | {quantiles(d)} | {b["mean"]:.2f} / {b["p99"]} / {b["max"]} | {m["mean"]:.2f} / {m["max"]} | {100*s["no_merge_pieces"]/s["pieces"]:.2f} |')
    lines += ["", "## Actual merge-map queries", "",
              "Hit = rule exists for the queried pair, NOT a BPE-cache hit and NOT necessarily an applied merge. Includes initial candidates, neighbour refreshes and heap revalidation. Frequencies include repeated queries, within and across pieces. Top 1% uses ceil(1% of unique queried pairs). Left>=64 is the fraction of all queries whose left ID has at least 64 merge rules, including queries that miss.", "",
              "| Model | Corpus | Queries | Queries/piece | Hit % | Unique pairs | Unique hit rules | Top 1% query share % | Queries with left>=64 % | Heap pieces % | Heap queries % |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in workloads:
        s = value["stats"]
        q = s["lookup"]["queries"]
        wide = sum(v for k, v in s["query_weighted_rules_per_left"]["histogram"].items() if int(k) >= 64)
        heap_q = sum(v["queries"] for k, v in s["lookup_sites"].items() if k.startswith("mergeHeap."))
        lines.append(f'| {value["model"]} | {value["corpus"]} | {q:,} | {q/s["pieces"]:.2f} | {100*s["lookup"]["hits"]/q:.2f} | {s["unique_queried_pairs"]:,} | {s["unique_hit_pairs"]:,} | {100*s["top_1_percent_pair_query_share"]:.2f} | {100*wide/q:.2f} | {100*s["paths"].get("heap",0)/s["pieces"]:.2f} | {100*heap_q/q:.2f} |')
    (output / "workload.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

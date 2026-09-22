#!/usr/bin/env python3
"""Prepare, validate and measure compact-v1; Python never enters a timed loop."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import tarfile
import tempfile

PROTOCOL = "compact-v1"
MODELS = ("gpt2", "qwen3-8")
LANGUAGES = ("en", "ru", "ar", "ja", "zh")
OPERATIONS = ("ModelTokenize", "TokenizerAppendIDs", "TokenizerEncode")
HARNESS = ("compact_benchmark_test.go", "corpus_benchmark_test.go")
WARMUP = 3
MEASUREMENT_FLAGS = ["-test.run=^$", "-test.bench=^BenchmarkCompact$", "-test.benchtime=1x",
                     "-test.count=1", "-test.cpu=1", "-test.benchmem"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def capture(command, cwd=None, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT).strip()


def optional(command):
    try:
        return capture(command, env=dict(os.environ, LC_ALL="C"))
    except (OSError, subprocess.CalledProcessError) as error:
        return "unavailable: " + str(error)


def host_info():
    info = dict(system=platform.system(), release=platform.release(), architecture=platform.machine(),
                logical_cpus=os.cpu_count(), affinity="not pinned")
    if platform.system() == "Darwin":
        info.update(cpu=optional(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    memory=optional(["sysctl", "-n", "hw.memsize"]),
                    os=optional(["sw_vers"]), isa=optional(["sysctl", "-a"]),
                    power=optional(["pmset", "-g", "batt"]), power_settings=optional(["pmset", "-g", "custom"]))
        # Retain CPU capabilities, not unrelated sysctl state or host identifiers.
        info["isa"] = "\n".join(line for line in info["isa"].splitlines()
                                if line.startswith(("hw.optional.", "machdep.cpu.features", "machdep.cpu.leaf7_features")))
    elif platform.system() == "Linux":
        info.update(lscpu=optional(["lscpu"]), os=platform.platform())
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in info["lscpu"].splitlines()
                            if line.startswith("Model name:")), platform.processor() or "unknown")
        info["affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unknown"
    else:
        raise ValueError("compact-v1 currently supports macOS and Linux")
    return info


def parse_samples(text, count):
    rows = {m + "/" + op: [] for m in MODELS for op in OPERATIONS}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("BenchmarkCompact/"):
            continue
        key = fields[0].removeprefix("BenchmarkCompact/").removesuffix("-1")
        if key not in rows or len(fields) < 4 or fields[1] != "1" or len(fields) % 2:
            raise ValueError("unexpected benchmark row: " + line)
        sample = {fields[i + 1]: float(fields[i]) for i in range(2, len(fields), 2)}
        required = ["ns/op", "B/op", "allocs/op", "input-bytes/op", "records/op", "id-sum/op"]
        required += [lang + "-4k-ns/op" for lang in LANGUAGES]
        if any(k not in sample or not math.isfinite(sample[k]) or sample[k] < 0 for k in required):
            raise ValueError("missing/nonfinite/negative metrics: " + key)
        if any(sample[k] <= 0 for k in ("ns/op", "input-bytes/op", "records/op")):
            raise ValueError("empty workload: " + key)
        rows[key].append(sample)
    for key, samples in rows.items():
        if len(samples) != count:
            raise ValueError(f"{key}: expected {count} samples, got {len(samples)}")
        for unit in ("input-bytes/op", "records/op", "id-sum/op"):
            if len({s[unit] for s in samples}) != 1:
                raise ValueError(f"{key}: workload changed between samples ({unit})")
    return rows


def median_interval(values):
    """Distribution-free >=95% order-statistic interval for the population median."""
    values = sorted(values)
    n, k = len(values), 0
    while k + 1 <= n // 2 and 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n <= .05:
        k += 1
    return [values[max(0, k - 1)], values[min(n - 1, n - k)]]


def ratio_interval(old, new):
    # Independent-sample bootstrap; explicitly not a paired experiment or a
    # correction for drift between historical runs. Fixed seed is reproducible.
    rng = random.Random(1729)
    values = sorted(statistics.median(rng.choices(old, k=len(old))) /
                    statistics.median(rng.choices(new, k=len(new))) for _ in range(4000))
    return dict(ratio=statistics.median(old) / statistics.median(new),
                ci95=[values[100], values[3899]], significant=values[100] > 1 or values[3899] < 1)


def summary(rows):
    out = {}
    for key, samples in rows.items():
        times = [s["ns/op"] for s in samples]
        size, records = samples[0]["input-bytes/op"], samples[0]["records/op"]
        lo, hi = median_interval(times)
        out[key] = dict(mb_s=1000 * size / statistics.median(times), mb_s_ci95=[1000 * size / hi, 1000 * size / lo],
                        ns_per_byte=statistics.median(times) / size,
                        bytes_per_record=statistics.median(s["B/op"] for s in samples) / records,
                        allocs_per_record=statistics.median(s["allocs/op"] for s in samples) / records,
                        records=int(records), input_bytes=int(size))
    return out


def public_metadata(meta):
    """Measurement evidence only; paths, shell commands and diagnostics stay local."""
    fields = ("protocol", "complete", "created_utc", "finished_utc", "commit", "machine",
              "source_modified", "go_version", "go_settings", "dependencies", "rust_version",
              "mode", "inputs", "workload_hash", "harness_hash", "source_hashes",
              "summary", "comparisons", "comparison_mode", "previous", "environment", "samples")
    result = {key: copy.deepcopy(meta[key]) for key in fields if key in meta}
    host = meta["host"]
    result["host"] = {key: copy.deepcopy(host[key]) for key in
                      ("cpu", "system", "release", "architecture", "logical_cpus", "affinity",
                       "memory", "os", "isa", "hypervisor", "virtualization") if key in host}
    for line in host.get("lscpu", "").splitlines():
        name, _, value = line.partition(":")
        key = {"Flags": "isa", "Hypervisor vendor": "hypervisor",
               "Virtualization type": "virtualization"}.get(name.strip())
        if key:
            result["host"][key] = value.strip()
    # Keep the power source, not battery device identifiers or diagnostic dumps.
    result["host"]["power"] = (host.get("power") or "unknown").splitlines()[0]
    result["process_environment"] = {key: meta.get("process_environment", {}).get(key, "")
                                     for key in ("GOWORK", "GOMAXPROCS", "GOGC", "GOMEMLIMIT", "GODEBUG",
                                                 "GOFLAGS", "GOEXPERIMENT", "RUSTUP_TOOLCHAIN",
                                                 "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")}
    if "previous" in result:
        result["previous"] = Path(result["previous"]).name
    result["measurement_flags"] = list(MEASUREMENT_FLAGS)
    result["throughput_unit"] = "MB/s (10^6 bytes/s)"
    return result


def publish_result(out, meta, samples, baseline_samples=None):
    """Export a completed run without copying any local execution artifacts."""
    if not meta.get("complete"):
        raise ValueError("cannot publish an incomplete run")
    rows = parse_samples(samples, meta["samples"])
    previous = None
    if baseline_samples is not None:
        # Historical baselines can have a different preselected sample count.
        count = sum(line.startswith("BenchmarkCompact/") for line in baseline_samples.splitlines())
        previous = parse_samples(baseline_samples, count // (len(MODELS) * len(OPERATIONS)))
    public = public_metadata(meta)
    report, result, comparisons = make_report(public, rows, previous)
    public.update(summary=result, comparisons=comparisons)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run.json", public)
    (out / "report.md").write_text(report)
    (out / "samples.txt").write_text(samples)
    if baseline_samples is not None:
        (out / "baseline-samples.txt").write_text(baseline_samples)
    return report


def check_comparable(current, previous):
    for field in ("protocol", "machine", "workload_hash", "harness_hash", "go_version", "go_settings", "dependencies"):
        if current[field] != previous[field]:
            raise ValueError(f"incomparable reports: {field} differs")
    for field in ("system", "release", "architecture", "cpu", "affinity"):
        if current["host"].get(field) != previous["host"].get(field):
            raise ValueError(f"incomparable host: {field} differs")
    for field in ("GOGC", "GOMEMLIMIT", "GODEBUG"):
        if current.get("process_environment", {}).get(field, "") != previous.get("process_environment", {}).get(field, ""):
            raise ValueError(f"incomparable runtime settings: {field} differs")
    if not previous.get("complete"):
        raise ValueError("previous report is incomplete")


def source_files(root):
    paths = capture(["git", "ls-files", "--cached", "--others", "--exclude-standard"], root).splitlines()
    return sorted(set(p for p in paths if not p.startswith(("testdata/generated/", "benchmarks/results/", "benchmarks/runs/", "tools/hf-reference/results/"))
                      and (p.endswith((".go", ".rs", ".py")) or p in ("go.mod", "go.sum", "Makefile", "benchmarks/suite.json", "tools/hf-reference/Cargo.toml", "tools/hf-reference/Cargo.lock", "tools/hf-reference/rust-toolchain.toml"))))


def prepare(root, run):
    generated = root / "testdata/generated/compact-v1"
    generated.mkdir(parents=True, exist_ok=True)
    # rustup honors RUSTUP_TOOLCHAIN; also reject a mismatched direct rustc on PATH.
    rust = run(["rustc", "--version"])
    if not rust.startswith("rustc 1.98.1 "):
        raise ValueError("install/select the pinned Rust 1.98.1 toolchain")
    run(["cargo", "build", "--locked", "--release", "--manifest-path", "tools/hf-reference/Cargo.toml", "--bin", "hf-reference", "--example", "prepare_nfc"])
    run(["go", "run", "./tools/benchmark-suite", "-models", ",".join(MODELS), "-prepare-only"])
    source_suite = json.loads((root / "benchmarks/suite.json").read_text())
    selected = [m for m in source_suite["models"] if m["name"] in MODELS]
    corpora = source_suite["corpora"]
    if [m["name"] for m in selected] != list(MODELS) or [c["name"] for c in corpora] != [x + "-4k" for x in LANGUAGES]:
        raise ValueError("source suite changed; update/version the compact protocol explicitly")
    inputs, prepared = {}, []
    for corpus in corpora:
        source = root / "testdata/generated/corpora" / (corpus["name"] + ".jsonl")
        dest = generated / source.name
        run(["tools/hf-reference/target/release/examples/prepare_nfc", str(source), str(dest)])
        before = [json.loads(x)["text"] for x in source.read_text().splitlines()]
        after = [json.loads(x)["text"] for x in dest.read_text().splitlines()]
        if len(before) != len(after):
            raise ValueError("normalization changed record count")
        inputs[corpus["name"]] = dict(source_sha256=sha(source), prepared_sha256=sha(dest),
            records=len(after), source_record_bytes=[len(s.encode()) for s in before],
            prepared_record_bytes=[len(s.encode()) for s in after],
            changed_records=sum(a != b for a, b in zip(before, after)))
        prepared.append(dict(name=corpus["name"], path=str(dest.relative_to(root))))
    models = []
    for model in selected:
        model = dict(model)
        directory = root / Path(model["tokenizer"]).parent
        # Verify upstream snapshots, not just the corpus-visible vocabulary.
        if (directory / "SHA256SUMS").exists():
            checksums = [line.split() for line in (directory / "SHA256SUMS").read_text().splitlines()]
        else:
            checksums = [(digest, name) for name, digest in re.findall(r"\| `([^`]+)` \| [\d,]+ \| `([0-9a-f]{64})`", (directory / "MANIFEST.md").read_text())]
            if len(checksums) != 3:
                raise ValueError("missing asset checksums")
        for expected, filename in checksums:
            if sha(directory / filename.lstrip("*")) != expected:
                raise ValueError("asset hash mismatch: " + filename)
        source = root / model["tokenizer"]
        value = json.loads(source.read_text())
        if model["name"] == "qwen3-8":
            if value.get("normalizer") != {"type": "NFC"}:
                raise ValueError("Qwen normalizer changed; do not silently remove it")
            value["normalizer"] = None
        elif value.get("normalizer") is not None:
            raise ValueError("unexpected normalizer")
        dest = generated / (model["name"] + "-tokenizer.json")
        write_json(dest, value)
        inputs[model["name"]] = dict(source_sha256=sha(source), prepared_sha256=sha(dest),
            vocab_sha256=sha(root / model["vocab"]), merges_sha256=sha(root / model["merges"]))
        model.update(tokenizer=str(dest.relative_to(root)), pipeline="json")
        models.append(model)
    suite = dict(schema=1, models=models, corpora=prepared)
    suite_path = generated / "suite.json"
    write_json(suite_path, suite)
    # A separate oracle namespace keeps extended-suite snapshots untouched.
    inputs["oracles"] = {}
    for model in models:
        for corpus in prepared:
            oracle = root / "testdata/generated/suite" / model["name"] / (corpus["name"] + "-compact.jsonl")
            oracle.parent.mkdir(parents=True, exist_ok=True)
            run(["tools/hf-reference/target/release/hf-reference", "generate", "--tokenizer", model["tokenizer"],
                 "--corpus", corpus["path"], "--output", str(oracle.relative_to(root))])
            inputs["oracles"][model["name"] + "/" + corpus["name"]] = sha(oracle)
    # ReferencePath derives its filename from corpus.name, not corpus.path.
    for corpus in suite["corpora"]:
        corpus["name"] += "-compact"
    write_json(suite_path, suite)
    inputs["corpus_manifest_sha256"] = sha(root / "testdata/corpora/text/MANIFEST.json")
    inputs["normalization"] = dict(form="NFC", per_record=True, implementation="HF tokenizers 0.23.2",
                                   cargo_lock_sha256=sha(root / "tools/hf-reference/Cargo.lock"))
    inputs["original_corpora"] = corpora
    inputs["source_model_manifests"] = {m["name"]: sha(root / Path(m["tokenizer"]).parent / "MANIFEST.md") for m in selected}
    return suite_path, inputs


def make_report(meta, rows, previous_rows=None):
    result = summary(rows)
    lines = ["# Compact BPE baseline — " + meta["machine"], "",
             f"UTC: {meta['created_utc']} · commit: `{meta['commit']}` · protocol: `{PROTOCOL}`", "",
             f"CPU: {meta['host']['cpu'].splitlines()[0]} · {meta['host']['system']} {meta['host']['release']} / {meta['host']['architecture']}",
             f"{meta['go_version']} · BPE backend: **scalar** (no SIMD dispatch implemented).",
             "pretok/unicode use their default dependency dispatch; this is not a forced-scalar full pipeline.",
             f"Single thread; {WARMUP} warmup passes before each sample; {meta['samples']} samples per row.",
             "Five NFC-prepared language pools (en/ru/ar/ja/zh), independently encoded ~4 KiB requests.",
             "Normalization, loading, compilation, allocation of dst, and HF correctness checks are outside timing.",
             "No BPE cache. Special-token insertion off; AddedToken matching on. AppendIDs reuses sufficient dst capacity.", "",
             "| Model | API | MB/s ↑ (95% median CI) | B/request ↓ | allocs/request ↓ | × previous | × scalar |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    comparisons = {}
    for key, row in result.items():
        model, operation = key.split("/")
        label = "GPT-2" if model == "gpt2" else "Qwen3.8 / prepared NFC"
        api = {"ModelTokenize": "Model.Tokenize", "TokenizerAppendIDs": "Tokenizer.AppendIDs", "TokenizerEncode": "Tokenizer.Encode"}[operation]
        ratio = "—"
        if previous_rows:
            old, new = previous_rows[key], rows[key]
            for unit in ("input-bytes/op", "records/op", "id-sum/op"):
                if old[0][unit] != new[0][unit]:
                    raise ValueError("comparison workload mismatch: " + key)
            comparison = ratio_interval([s["ns/op"] for s in old], [s["ns/op"] for s in new])
            comparisons[key] = comparison
            ratio = f"{comparison['ratio']:.3g}×" + ("" if comparison["significant"] else " (noise)")
        lo, hi = row["mb_s_ci95"]
        lines.append(f"| {label} | {api} | {row['mb_s']:.3g} [{lo:.3g}, {hi:.3g}] | {row['bytes_per_record']:.0f} | {row['allocs_per_record']:.2f} | {ratio} | — |")
    lines += ["", "1 MB = 10⁶ bytes; MB/s = 1000 × input bytes / nanoseconds. Model counts transformed-piece UTF-8 bytes; Tokenizer counts prepared source bytes.",
              "One timed op is a complete five-language pass; allocations are divided by source-record count, including for Model.",
              "Timed paths consume every output ID in a checksum. Language timing boundaries add five clock intervals per pass.",
              "CI: distribution-free order-statistic interval for the median. Ratios, when present, use an independent-sample bootstrap (4,000 resamples, seed 1729).",
              "Scalar comparison is unavailable: both current production merge paths are scalar; no heap-oracle comparison is substituted.", "",
              "## Execution conditions", "", meta["environment"], "",
              "Affinity: " + str(meta["host"]["affinity"]) + ". Power source: " + (meta["host"].get("power") or "unknown").splitlines()[0] + ".", "",
              "Raw samples: [samples.txt](samples.txt). Measurement metadata: [run.json](run.json).",
              "HF 0.23.2 is a correctness oracle here, not a timed competitor."]
    if meta.get("source_modified"):
        lines += ["", "Measured source includes uncommitted changes; its file hashes are recorded in run.json."]
    if previous_rows:
        lines += ["", "## Previous comparison", "", meta["comparison_mode"], "Baseline: " + meta["previous"]]
        regressions = []
        for key in rows:
            for lang in LANGUAGES:
                unit = lang + "-4k-ns/op"
                change = ratio_interval([s[unit] for s in previous_rows[key]], [s[unit] for s in rows[key]])
                if change["ci95"][1] < 1:
                    regressions.append(f"- {key}/{lang}: {change['ratio']:.3g}×")
        lines += ["", "Language regressions (exploratory, no multiple-comparison correction):"] + (regressions or ["None detected."])
    else:
        lines += ["", "First baseline: no previous version was supplied; no speedup is claimed."]
    return "\n".join(lines) + "\n", result, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, help="CPU/ISA label, e.g. apple-m5 or xeon-skylake-avx512; not an SSH alias")
    parser.add_argument("--output", type=Path, help="new published result directory; local diagnostics stay in ignored benchmarks/runs")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--environment", default="Background load and thermal conditions uncontrolled; no CPU affinity or power changes requested.",
                        help="public execution conditions, without host aliases, local paths or setup instructions")
    parser.add_argument("--previous", type=Path, help="historical report directory (not a same-run A/B)")
    parser.add_argument("--baseline-ref", help="git ref with the identical compact harness; alternate fresh runs")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.machine) or args.samples < 10:
        parser.error("machine must be a lowercase slug; samples must be >=10")
    if args.previous and args.baseline_ref:
        parser.error("choose --previous or --baseline-ref")
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "testdata/generated/compact-v1/run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_benchmark(root, args)


def run_benchmark(root, args):
    env = dict(os.environ, GOWORK="off", GOMAXPROCS="1")
    env["LC_ALL"] = "C"
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    toolchain = (root / "tools/hf-reference/rust-toolchain.toml").read_text()
    env["RUSTUP_TOOLCHAIN"] = re.search(r'channel\s*=\s*"([^"]+)"', toolchain).group(1)
    env["RAYON_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("BPE_BENCH_SUITE", None)
    env.pop("BPE_BENCH_MODELS", None)
    env.pop("BPE_BENCH_CORPORA", None)
    commands = []

    def run(command, cwd=root, extra=None):
        command = list(map(str, command))
        commands.append(dict(argv=command, cwd=str(cwd)))
        print("+ " + " ".join(command), flush=True)
        return capture(command, cwd, dict(env, **(extra or {})))

    suite, inputs = prepare(root, run)
    if args.prepare_only:
        print(suite)
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    commit = capture(["git", "rev-parse", "HEAD"], root)
    out = (args.output or root / "benchmarks/results/compact-v1" / args.machine / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:8])).resolve()
    if out.exists():
        raise FileExistsError(out)
    local = root / "benchmarks/runs/compact-v1" / args.machine / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:8])
    local.mkdir(parents=True, exist_ok=False)
    metadata = dict(protocol=PROTOCOL, complete=False, created_utc=now.isoformat(), commit=commit,
        machine=args.machine, host=host_info(), environment=args.environment, samples=args.samples,
        go_version=capture(["go", "version"], root, env),
        go_settings=json.loads(capture(["go", "env", "-json", "GOOS", "GOARCH", "GOEXPERIMENT", "GOAMD64", "GOARM64", "GOFLAGS", "CGO_ENABLED"], root, env)),
        dependencies=capture(["go", "list", "-m", "all"], root, env),
        rust_version=capture(["rustc", "--version"], root / "tools/hf-reference", env),
        mode=dict(warmup=WARMUP, threads=1, bpe_backend="scalar", scalar_comparison="unavailable", add_special_tokens=False,
                  dst="reused, capacity=max expected IDs/record", corpus="NFC per record, fixed language order"),
        inputs=inputs, workload_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        harness_hash={p: sha(root / p) for p in HARNESS},
        source_hashes={p: sha(root / p) for p in source_files(root)}, commands=commands)
    metadata["process_environment"] = {key: env.get(key, "") for key in
        ("GOWORK", "GOMAXPROCS", "GOGC", "GOMEMLIMIT", "GODEBUG", "GOFLAGS", "GOEXPERIMENT", "GOCACHE", "CARGO_HOME", "RUSTUP_TOOLCHAIN", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PATH")}
    # Save a replayable patch including new harness files, never generated data/results.
    tracked = capture(["git", "ls-files"], root).splitlines()
    changes = run(["git", "diff", "HEAD", "--", *metadata["source_hashes"]]) + "\n"
    for path in metadata["source_hashes"]:
        if path not in tracked:
            diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", path], cwd=root, text=True, stdout=subprocess.PIPE)
            if diff.returncode not in (0, 1):
                raise ValueError("cannot snapshot " + path)
            changes += diff.stdout
    (local / "source.patch").write_text(changes)
    metadata["source_patch_sha256"] = sha(local / "source.patch")
    metadata["source_modified"] = bool(changes.strip())
    write_json(local / "run.json", metadata)
    previous_rows = None
    baseline_samples = None
    if args.previous:
        old = json.loads((args.previous / "run.json").read_text())
        check_comparable(metadata, old)
        baseline_samples = (args.previous / "samples.txt").read_text()
        previous_rows = parse_samples(baseline_samples, old["samples"])
        metadata.update(previous=old["commit"], comparison_mode="Historical comparison; thermal/power/background drift is not controlled by statistical intervals.")
    try:
        binary = root / "testdata/generated/compact-v1/compact.test"
        run(["go", "test", "-mod=readonly", "-c", "-o", binary, "."])
        extra = dict(BPE_COMPACT_SUITE=str(suite), BPE_HF_REPOSITORY=str(root))
        validation = run([binary, "-test.run=^TestHFSuite$", "-test.count=1"], extra=dict(BPE_BENCH_SUITE=str(suite), BPE_HF_REPOSITORY=str(root)))
        (local / "validation.txt").write_text(validation + "\n")
        metadata["binary_sha256"] = sha(binary)
        binaries = {"current": binary}
        baseline_temp = None
        if args.baseline_ref:
            ref = run(["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"])
            baseline_temp = tempfile.TemporaryDirectory(prefix="bpe-compact-baseline-")
            checkout = Path(baseline_temp.name)
            archive = checkout / "source.tar"
            with archive.open("wb") as stream:
                subprocess.run(["git", "archive", ref], cwd=root, stdout=stream, check=True)
            with tarfile.open(archive) as stream:
                for member in stream.getmembers():
                    if not (member.isfile() or member.isdir()) or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                        raise ValueError("unsupported baseline archive member: " + member.name)
                stream.extractall(checkout)
            for path in HARNESS:
                if not (checkout / path).exists() or sha(checkout / path) != metadata["harness_hash"][path]:
                    raise ValueError("baseline ref must contain identical compact-v1 harness: " + path)
            if (checkout / "go.mod").read_bytes() != (root / "go.mod").read_bytes() or (checkout / "go.sum").read_bytes() != (root / "go.sum").read_bytes():
                raise ValueError("same-run A/B requires identical dependencies")
            binaries["baseline"] = checkout / "compact.test"
            run(["go", "test", "-mod=readonly", "-c", "-o", binaries["baseline"], "."], cwd=checkout)
            run([binaries["baseline"], "-test.run=^TestHFSuite$", "-test.count=1"], extra=dict(BPE_BENCH_SUITE=str(suite), BPE_HF_REPOSITORY=str(root)))
            metadata.update(previous=ref, comparison_mode="Same-run A/B: process order alternates each sample; identical toolchain, dependencies, inputs and measurement harness.", baseline_binary_sha256=sha(binaries["baseline"]))
        logs = {key: "" for key in binaries}
        for sample in range(args.samples):
            order = list(binaries)
            if sample % 2:
                order.reverse()
            for name in order:
                text = run([binaries[name], *MEASUREMENT_FLAGS], extra=extra)
                parse_samples(text, 1)
                logs[name] += f"# sample {sample + 1} {name}\n{text}\n"
                (local / ("samples.txt" if name == "current" else "baseline-samples.txt")).write_text(logs[name])
                print(f"sample {sample+1}/{args.samples} {name}: complete", flush=True)
        rows = parse_samples(logs["current"], args.samples)
        if "baseline" in logs:
            baseline_samples = logs["baseline"]
            previous_rows = parse_samples(baseline_samples, args.samples)
        if set(source_files(root)) != set(metadata["source_hashes"]):
            raise ValueError("source file set changed during run")
        for path, digest in metadata["source_hashes"].items():
            if sha(root / path) != digest:
                raise ValueError("source changed during run: " + path)
        report, result, comparisons = make_report(metadata, rows, previous_rows)
        metadata.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        summary=result, comparisons=comparisons, power_end=optional(["pmset", "-g", "batt"]) if platform.system() == "Darwin" else "unknown")
        write_json(local / "run.json", metadata)
        report = publish_result(out, metadata, logs["current"], baseline_samples)
        if baseline_temp:
            baseline_temp.cleanup()
        print(report)
        print("Report: " + str(out / "report.md"))
    except Exception as error:
        metadata["error"] = str(error)
        write_json(local / "run.json", metadata)
        raise


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.output or "", flush=True)
        raise

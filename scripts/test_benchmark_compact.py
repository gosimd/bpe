import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmark_compact as bench


def raw_samples(count=20):
    text = ""
    for model in bench.MODELS:
        for op in bench.OPERATIONS:
            for _ in range(count):
                text += f"BenchmarkCompact/{model}/{op} 1 2000 ns/op 1000 input-bytes/op 10 records/op 7 id-sum/op 50 B/op 5 allocs/op"
                text += "".join(f" 300 {lang}-4k-ns/op" for lang in bench.LANGUAGES) + "\n"
    return text


class CompactTests(unittest.TestCase):
    def test_six_rows_and_units(self):
        rows = bench.parse_samples(raw_samples(), 20)
        self.assertEqual(len(rows), 6)
        for row in bench.summary(rows).values():
            self.assertEqual(row["mb_s"], 500)
            self.assertEqual(row["mb_s_ci95"], [500, 500])
            self.assertEqual(row["ns_per_byte"], 2)
            self.assertEqual(row["bytes_per_record"], 5)
            self.assertEqual(row["allocs_per_record"], .5)

    def test_refuse_incomplete_or_changed_work(self):
        for text in (raw_samples(19), raw_samples().replace("1000 input-bytes/op", "2000 input-bytes/op", 1),
                     raw_samples().replace("2000 ns/op", "nan ns/op", 1),
                     raw_samples().replace(" 1 2000", " 2 2000", 1)):
            with self.assertRaises(ValueError):
                bench.parse_samples(text, 20)

    def test_comparison_direction_and_noise(self):
        self.assertEqual(bench.ratio_interval([2000] * 20, [1000] * 20)["ratio"], 2)
        self.assertTrue(bench.ratio_interval([2000] * 20, [1000] * 20)["significant"])
        self.assertFalse(bench.ratio_interval([1000] * 20, [1000] * 20)["significant"])
        self.assertEqual(bench.median_interval(list(range(1, 21))), [6, 15])

    def test_incompatible_reports_fail(self):
        current = {key: "same" for key in ("protocol", "machine", "workload_hash", "harness_hash", "go_version", "go_settings", "dependencies")}
        current.update(host=dict(cpu="M5"), complete=True)
        bench.check_comparable(current, current)
        for field in ("workload_hash", "machine", "dependencies"):
            previous = copy.deepcopy(current)
            previous[field] = "different"
            with self.assertRaises(ValueError):
                bench.check_comparable(current, previous)
        previous = copy.deepcopy(current)
        previous["complete"] = False
        with self.assertRaises(ValueError):
            bench.check_comparable(current, previous)

    def test_baseline_has_no_fabricated_speedups(self):
        rows = bench.parse_samples(raw_samples(), 20)
        metadata = dict(machine="test", created_utc="test", commit="abc", samples=20,
                        host=dict(cpu="CPU", system="Linux", release="test", architecture="amd64", affinity="unknown"),
                        go_version="go1.27.0", environment="uncontrolled")
        report, result, comparisons = bench.make_report(metadata, rows)
        self.assertEqual(len(result), 6)
        self.assertEqual(report.count("| — | — |"), 6)
        self.assertEqual(comparisons, {})
        self.assertIn("MB/s ↑", report)
        self.assertNotIn("GB/s", report)
        self.assertNotIn("source.patch", report)

    def test_publication_keeps_only_measurements(self):
        metadata = dict(protocol=bench.PROTOCOL, machine="xeon-skylake-avx512", created_utc="test",
                        commit="abc", samples=20, complete=True, source_modified=True,
                        host=dict(cpu="Xeon", system="Linux", release="test", architecture="amd64",
                                  affinity=[1], lscpu="Flags: avx2 avx512f\nHypervisor vendor: KVM\n",
                                  power="Battery Power\nInternalBattery id=123"),
                        go_version="go1.27.0", environment="KVM guest",
                        process_environment=dict(PATH="/private/bin", GOCACHE="/private/cache", GOGC="100"),
                        commands=[dict(cwd="/private/checkout", argv=["ssh", "private-host"])],
                        source_patch_sha256="local", error="private diagnostics")
        before = copy.deepcopy(metadata)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "published"
            bench.publish_result(out, metadata, raw_samples())
            self.assertEqual({p.name for p in out.iterdir()}, {"report.md", "run.json", "samples.txt"})
            self.assertEqual((out / "samples.txt").read_text(), raw_samples())
            public = json.loads((out / "run.json").read_text())
            self.assertEqual(public["host"]["hypervisor"], "KVM")
            self.assertEqual(public["host"]["isa"], "avx2 avx512f")
            self.assertEqual(public["process_environment"]["GOGC"], "100")
            self.assertEqual(public["measurement_flags"], bench.MEASUREMENT_FLAGS)
            self.assertEqual(bench.public_metadata(public), public)
            for word in ("private", "commands", "source_patch", "InternalBattery", "gb_s"):
                self.assertNotIn(word, (out / "run.json").read_text())
            with self.assertRaises(FileExistsError):
                bench.publish_result(out, metadata, raw_samples())
        self.assertEqual(metadata, before)

    def test_incomplete_publication_fails_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "incomplete"
            with self.assertRaises(ValueError):
                bench.publish_result(out, dict(complete=False), raw_samples())
            self.assertFalse(out.exists())

    def test_published_baseline_comparison(self):
        metadata = dict(protocol=bench.PROTOCOL, machine="apple-m5", created_utc="test", commit="abc",
                        samples=20, complete=True, go_version="go1.27.0", environment="uncontrolled",
                        host=dict(cpu="M5", system="Darwin", release="test", architecture="arm64", affinity="not pinned"),
                        previous="/private/older-run", comparison_mode="Historical comparison")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "comparison"
            bench.publish_result(out, metadata, raw_samples(), raw_samples(10))
            self.assertEqual({p.name for p in out.iterdir()},
                             {"report.md", "run.json", "samples.txt", "baseline-samples.txt"})
            public = json.loads((out / "run.json").read_text())
            self.assertEqual(public["previous"], "older-run")
            self.assertEqual(len(public["comparisons"]), 6)
            self.assertNotIn("/private", (out / "report.md").read_text())

    def test_runner_separates_local_and_published_artifacts(self):
        # Exercise orchestration without compiling Rust or running timed work.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toolchain = root / "tools/hf-reference/rust-toolchain.toml"
            toolchain.parent.mkdir(parents=True)
            toolchain.write_text('[toolchain]\nchannel = "1.98.1"\n')
            published = root / "published"

            def capture(command, cwd=None, env=None):
                if command[:2] == ["git", "rev-parse"]:
                    return "abc123"
                if command[0] == "git":
                    return ""
                if command[:2] == ["go", "version"]:
                    return "go version go1.27.0 linux/amd64"
                if command[:2] == ["go", "env"]:
                    return '{"GOARCH":"amd64"}'
                if command[:2] == ["go", "list"]:
                    return "github.com/gosimd/bpe"
                if command[0] == "rustc":
                    return "rustc 1.98.1"
                if command[:2] == ["go", "test"]:
                    Path(command[command.index("-o") + 1]).write_bytes(b"test binary")
                    return ""
                if "-test.run=^TestHFSuite$" in command:
                    return "PASS"
                if "-test.bench=^BenchmarkCompact$" in command:
                    return raw_samples(1)
                self.fail(f"unexpected command: {command}")

            host = dict(cpu="Test CPU", system="Linux", release="test", architecture="amd64", affinity=[1])
            with mock.patch.multiple(bench, capture=capture, host_info=lambda: host,
                                     prepare=lambda root, run: (root / "suite.json", {}),
                                     source_files=lambda root: [], optional=lambda command: "unknown", HARNESS=()), \
                    mock.patch.object(bench, "__file__", str(root / "scripts/benchmark_compact.py")), \
                    mock.patch("sys.argv", ["benchmark_compact.py", "--machine", "test-cpu", "--samples", "10",
                                            "--output", str(published)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            self.assertEqual({p.name for p in published.iterdir()}, {"report.md", "run.json", "samples.txt"})
            local, = (root / "benchmarks/runs/compact-v1/test-cpu").iterdir()
            self.assertTrue((local / "source.patch").exists())
            self.assertEqual((local / "validation.txt").read_text(), "PASS\n")
            self.assertIn("commands", json.loads((local / "run.json").read_text()))
            self.assertNotIn("commands", json.loads((published / "run.json").read_text()))
            bench.parse_samples((published / "samples.txt").read_text(), 10)


if __name__ == "__main__":
    unittest.main()

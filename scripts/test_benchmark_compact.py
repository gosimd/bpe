import copy
import unittest

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
            self.assertEqual(row["gb_s"], .5)
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


if __name__ == "__main__":
    unittest.main()

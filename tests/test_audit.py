import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.metrics import cohen_kappa_score

from movie_rec.audit import (
    cohen_kappa,
    sample_explanations,
    sample_triples,
    score_explanations,
    score_triples,
    triple_coverage,
)


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def fill(path, values):
    rows = read(path)
    for row in rows:
        row.update(values(row))
    write(Path(path), rows[0].keys(), [r.values() for r in rows])


class KappaTests(unittest.TestCase):
    def test_matches_sklearn(self):
        a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        b = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
        self.assertAlmostEqual(cohen_kappa(a, b), cohen_kappa_score(a, b))
        x = [1, 2, 3, 4, 5, 5, 4, 3]
        y = [1, 3, 3, 5, 5, 4, 4, 2]
        self.assertAlmostEqual(
            cohen_kappa(x, y, labels=range(1, 6), weights="linear"),
            cohen_kappa_score(x, y, labels=list(range(1, 6)), weights="linear"),
        )

    def test_identical_constant_ratings(self):
        self.assertEqual(cohen_kappa([1, 1, 1], [1, 1, 1]), 1.0)


class TripleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = self.root = Path(self.temporary.name)
        write(root / "catalog.csv", ("movie_id", "title", "overview"),
              [(m, f"Movie {m}", f"Overview {m}") for m in range(1, 6)])
        write(root / "semantic_triples.csv",
              ("movie_id", "tmdb_id", "predicate", "normalized_label", "evidence", "model", "prompt_version"),
              [(m, m * 10, "HAS_THEME", f"theme {i}", "q", "qwen2.5:3b", "abc")
               for m in range(1, 5) for i in range(3)])
        write(root / "extraction_status.csv", ("movie_id", "status"),
              [(1, "ok"), (2, "ok"), (3, "ok"), (4, "ok"), (5, "extraction_failed")])

    def test_sample_and_score(self):
        out = self.root / "audit"
        paths = sample_triples(self.root / "semantic_triples.csv", self.root / "catalog.csv", out, n=10)
        sheet = read(paths["A"])
        self.assertEqual(len(sheet), 10)
        self.assertTrue(all(row["title"].startswith("Movie") for row in sheet))
        self.assertEqual(read(paths["A"]), read(paths["B"]))
        fill(paths["A"], lambda r: {"correct": "1", "relevant": "1", "ontology_consistent": "1"})
        fill(paths["B"], lambda r: {
            "correct": "0" if r["audit_id"] == "1" else "yes", "relevant": "1", "ontology_consistent": "y",
        })
        summary = score_triples(
            paths["A"], paths["B"], out,
            self.root / "extraction_status.csv", self.root / "semantic_triples.csv",
        )
        self.assertEqual(summary["n_triples"], 10)
        self.assertAlmostEqual(summary["criteria"]["valid"]["rate_mean"], 0.95)
        self.assertTrue(summary["met_valid_target"])
        self.assertEqual(summary["disagreements"], ["1"])
        self.assertAlmostEqual(summary["coverage"]["coverage"], 0.8)
        self.assertFalse(summary["coverage"]["met_target"])
        self.assertTrue((out / "triple_audit_summary.json").exists())

    def test_blank_mark_is_an_error(self):
        paths = sample_triples(self.root / "semantic_triples.csv", self.root / "catalog.csv",
                               self.root / "audit", n=3)
        with self.assertRaisesRegex(ValueError, "must be 1 or 0"):
            score_triples(paths["A"], paths["B"])

    def test_coverage_ignores_missing_overviews(self):
        write(self.root / "status2.csv", ("movie_id", "status"),
              [(1, "ok"), (9, "missing_overview")])
        coverage = triple_coverage(self.root / "status2.csv", self.root / "semantic_triples.csv")
        self.assertEqual(coverage["n_processed"], 1)
        self.assertEqual(coverage["coverage"], 1.0)


class ExplanationAuditTests(unittest.TestCase):
    def test_sample_and_score(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "catalog.csv", ("movie_id", "title", "overview"),
                  [(m, f"Movie {m}", "x") for m in range(1, 11)])
            header = ("seed_movie_id", "rank", "movie_id", "score", "explanation")
            write(root / "meta.csv", header,
                  [(s, 1, s % 10 + 1, 1.0, "shares genre Drama") for s in range(1, 11)])
            write(root / "llm.csv", header,
                  [(s, 1, s % 10 + 1, 1.0, "shares theme revenge") for s in range(1, 11)])
            out = root / "audit"
            paths = sample_explanations(
                {"metadata_kg": root / "meta.csv", "llm_kg": root / "llm.csv"},
                root / "catalog.csv", out, n=8,
            )
            sheet = read(paths["A"])
            self.assertEqual(len(sheet), 16)
            self.assertNotIn("system", sheet[0])
            key = {row["item_id"]: row["system"] for row in read(paths["key"])}

            def scores(offset):
                return lambda r: {
                    "accuracy": str(min(5, 4 + offset)) if key[r["item_id"]] == "llm_kg" else "3",
                    "usefulness": "4" if key[r["item_id"]] == "llm_kg" else str(2 + offset),
                }
            fill(paths["A"], scores(0))
            fill(paths["B"], scores(1))
            summary = score_explanations(paths["A"], paths["B"], paths["key"], out)
            self.assertAlmostEqual(summary["systems"]["llm_kg"]["accuracy"]["mean"], 4.5)
            self.assertTrue(summary["systems"]["llm_kg"]["accuracy"]["met_target"])
            self.assertAlmostEqual(summary["systems"]["metadata_kg"]["usefulness"]["mean"], 2.5)
            self.assertEqual(summary["tests"][0]["system_a"], "llm_kg")
            self.assertEqual(summary["tests"][0]["n_seeds"], 8)
            self.assertLess(summary["tests"][0]["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()

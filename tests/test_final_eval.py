import csv
import gzip
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from movie_rec.final_eval import (
    RankingFileRecommender,
    _query_metrics,
    holm_adjust,
    paired_wilcoxon,
    parse_system_specs,
    run_final_evaluation,
)
from movie_rec.metrics import ranking_metrics

MOVIES = range(1, 31)


def write_catalog(path):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "title", "overview", "genres", "keywords"))
        for movie_id in MOVIES:
            writer.writerow((movie_id, f"Movie {movie_id}", "plot", "[]", "[]"))


def write_ratings(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        writer.writerows(rows)


def write_rankings(path, lists, explanations=False):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        header = ["seed_movie_id", "rank", "movie_id", "score"]
        if explanations:
            header.append("explanation")
        writer.writerow(header)
        for seed, movies in lists.items():
            for rank, movie in enumerate(movies, 1):
                row = [seed, rank, movie, 1.0 / rank]
                if explanations:
                    row.append(f"Movie {seed} -HAS_THEME-> theme <-HAS_THEME- Movie {movie}")
                writer.writerow(row)


def ordered(seed, first=()):
    """Twenty candidates for a seed, starting with `first`."""
    first = [m for m in first if m != seed]
    rest = [m for m in MOVIES if m != seed and m not in first]
    return (first + rest)[:20]


class FastMetricTests(unittest.TestCase):
    def test_matches_member2_metrics(self):
        rng = random.Random(7)
        ks = (5, 10, 20)
        idcg = [0.0]
        import math
        for rank in range(1, 21):
            idcg.append(idcg[-1] + 1.0 / math.log2(rank + 1))
        for _ in range(300):
            ranked = rng.sample(range(100), rng.randint(0, 20))
            relevant = set(rng.sample(range(100), rng.randint(1, 30)))
            fast = _query_metrics(ranked, relevant, ks, idcg)
            for i, k in enumerate(ks):
                expected = ranking_metrics(ranked, relevant, k)
                for j, metric in enumerate(("precision", "recall", "ndcg", "hit_rate")):
                    self.assertAlmostEqual(fast[i, j], expected[metric], places=12)


class StatisticsTests(unittest.TestCase):
    def test_holm(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.06)

    def test_all_ties_give_p_one(self):
        statistic, p_value, n_nonzero = paired_wilcoxon([0.5, 0.2], [0.5, 0.2])
        self.assertIsNone(statistic)
        self.assertEqual(p_value, 1.0)
        self.assertEqual(n_nonzero, 0)

    def test_clear_difference_is_significant(self):
        rng = np.random.default_rng(0)
        base = rng.random(200)
        _, p_value, _ = paired_wilcoxon(base + 0.1, base)
        self.assertLess(p_value, 1e-10)

    def test_system_specs(self):
        self.assertEqual(list(parse_system_specs(["a=x.csv", "b = y.npz"])), ["a", "b"])
        with self.assertRaises(ValueError):
            parse_system_specs(["broken"])
        with self.assertRaises(ValueError):
            parse_system_specs(["a=x", "a=y"])


class RankingFileTests(unittest.TestCase):
    def test_reads_sorted_lists_and_explanations(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "r.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(("seed_movie_id", "rank", "movie_id", "score", "explanation"))
                writer.writerow((1, 2, 3, 0.5, "b"))
                writer.writerow((1, 1, 2, 0.9, "a"))
            model = RankingFileRecommender(path)
            self.assertEqual([m for m, _, _ in model.recommend(1, 5)], [2, 3])
            self.assertEqual(model.explanation(1, 3), "b")
            self.assertEqual(model.recommend(99, 5), [])

    def test_rejects_self_recommendation_and_duplicate_rank(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "r.csv"
            for rows in (((1, 1, 1),), ((1, 1, 2), (1, 1, 3))):
                with path.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(("seed_movie_id", "rank", "movie_id"))
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    RankingFileRecommender(path)


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = self.root = Path(self.temporary.name)
        write_catalog(root / "catalog.csv")
        # Movie 1 is popular (12 training ratings); movie 2 is cold (2 ratings).
        train = [(user, 1, 5.0, 1) for user in range(1, 13)]
        train += [(user, 2, 5.0, 2) for user in (1, 2)]
        write_ratings(root / "train.csv.gz", train)
        # Users 1 and 2 like 5 in test; user 3 likes 6.
        write_ratings(root / "test.csv.gz", [
            (1, 5, 5.0, 10), (2, 5, 4.5, 10), (3, 6, 4.0, 10), (4, 7, 2.0, 10),
        ])
        (root / "manifest.json").write_text(json.dumps({"positive_rating_threshold": 4.0}))
        self.good = root / "llm_kg.csv"
        self.weak = root / "tfidf.csv"
        write_rankings(self.good, {s: ordered(s, (5, 6)) for s in MOVIES})
        write_rankings(self.weak, {s: ordered(s, ()) [::-1] for s in MOVIES})

    def test_end_to_end(self):
        output = self.root / "out"
        results, tests, hypotheses, paths = run_final_evaluation(
            {"tfidf": self.weak, "llm_kg": self.good}, self.root, output,
        )
        # Queries: users 1,2 seed {1,2}; user 3 seed {1}. 5 queries, 2 cold (seed 2).
        good = results["llm_kg"]["summary"]
        self.assertEqual(good["all"]["n_queries"], 5)
        self.assertEqual(good["cold_start"]["n_queries"], 2)
        self.assertEqual(good["all"]["metrics"][5]["hit_rate"], 1.0)
        self.assertEqual(good["all"]["coverage"], 1.0)
        self.assertGreater(
            good["all"]["metrics"][10]["ndcg"],
            results["tfidf"]["summary"]["all"]["metrics"][10]["ndcg"],
        )
        self.assertEqual(len(tests), 1)
        self.assertEqual((tests[0]["system_a"], tests[0]["system_b"]), ("llm_kg", "tfidf"))
        self.assertEqual(hypotheses[0]["metric"], "ndcg@10")
        for key in ("metrics_csv", "significance_csv", "metrics_json", "report_md"):
            self.assertTrue(paths[key].exists())
        with paths["metrics_csv"].open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2 * 2 * 3)
        self.assertIn("Cold-start", paths["report_md"].read_text(encoding="utf-8"))

    def test_missing_and_short_lists_are_reported(self):
        partial = self.root / "partial.csv"
        lists = {1: ordered(1, (5,)), 2: ordered(2)[:3]}
        write_rankings(partial, lists)
        results, _, _, paths = run_final_evaluation({"kg": partial}, self.root, self.root / "out")
        stats = results["kg"]["list_stats"]
        # Query seeds are 1 and 2: seed 2 is short, nothing is missing.
        self.assertEqual((stats["n_seeds"], stats["n_missing"], stats["n_short"]), (2, 0, 1))
        self.assertIn("Ranking-list completeness", paths["report_md"].read_text(encoding="utf-8"))
        # A seed with no list scores zero instead of stopping the run.
        write_rankings(partial, {1: ordered(1, (5,))})
        results, _, _, _ = run_final_evaluation({"kg": partial}, self.root, self.root / "out")
        self.assertEqual(results["kg"]["list_stats"]["n_missing"], 1)
        self.assertEqual(results["kg"]["summary"]["cold_start"]["metrics"][20]["hit_rate"], 0.0)

    def test_files_ranked_below_max_k_are_an_error(self):
        short = self.root / "short.csv"
        write_rankings(short, {s: ordered(s)[:10] for s in MOVIES})
        with self.assertRaisesRegex(ValueError, "--k 20"):
            run_final_evaluation({"short": short}, self.root, self.root / "out")

    def test_cli(self):
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo / "src")}
        completed = subprocess.run(
            [
                sys.executable, "-m", "movie_rec.cli", "final-evaluate",
                "--system", f"tfidf={self.weak}", "--system", f"llm_kg={self.good}",
                "--directory", str(self.root), "--catalog", str(self.root / "catalog.csv"),
                "--output-dir", str(self.root / "cli"), "--write-query-metrics",
            ],
            cwd=repo, env=env, check=True, capture_output=True, text=True,
        )
        self.assertIn("Final evaluation", completed.stdout)
        self.assertTrue((self.root / "cli" / "final_query_primary_metric.csv.gz").exists())


if __name__ == "__main__":
    unittest.main()

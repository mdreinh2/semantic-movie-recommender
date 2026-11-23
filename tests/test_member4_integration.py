"""Member 4 KG rankings flow into the Member 5 evaluation and audits."""

import csv
import gzip
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from movie_rec.audit import sample_explanations
from movie_rec.final_eval import run_final_evaluation

ROOT = Path(__file__).resolve().parents[1]
GENRES = (["Animation", "Family"], ["Crime"], ["Drama"], ["Action"])


def write_catalog(path):
    columns = ("movie_id", "tmdb_id", "title", "overview", "release_date", "genres",
               "keywords", "cast", "directors", "train_count", "popularity_band")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(columns)
        for movie_id in range(1, 41):
            writer.writerow((
                movie_id, movie_id * 10, f"Movie {movie_id}", "plot", "",
                json.dumps(GENRES[movie_id % 4]), json.dumps([f"kw{movie_id % 6}"]),
                json.dumps([f"Actor {movie_id % 5}"]), json.dumps([f"Director {movie_id % 7}"]),
                0, "0",
            ))


def write_triples(path):
    # Member 3's semantic_triples.csv format.
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "tmdb_id", "predicate", "normalized_label",
                         "evidence", "model", "prompt_version"))
        for movie_id in range(1, 41):
            writer.writerow((movie_id, movie_id * 10, "HAS_THEME", f"theme {movie_id % 3}",
                             "q", "qwen2.5:3b", "abc"))


def write_ratings(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        writer.writerows(rows)


class Member4IntegrationTests(unittest.TestCase):
    def test_kg_rankings_evaluate_and_audit(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_catalog(root / "catalog.csv")
            write_triples(root / "semantic_triples.csv")
            write_ratings(root / "train.csv.gz",
                          [(u, m, 5.0, i) for u in range(1, 11) for i, m in enumerate(range(u, u + 6))])
            write_ratings(root / "test.csv.gz",
                          [(u, m, 4.5, 100 + i) for u in range(1, 11) for i, m in enumerate(range(u + 20, u + 24))])
            files = {}
            for name, mode in (("metadata_kg", "metadata"), ("llm_kg", "enhanced")):
                files[name] = root / f"{name}_rankings.csv"
                command = [sys.executable, "-m", "movie_rec.kg.cli", "rank-catalog",
                           "--catalog", str(root / "catalog.csv"), "--mode", mode,
                           "--backend", "memory", "--k", "20", "--output", str(files[name])]
                if mode == "enhanced":
                    command += ["--triples", str(root / "semantic_triples.csv")]
                subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)

            results, tests, _, paths = run_final_evaluation(files, root, root / "member5")
            self.assertEqual(set(results), {"metadata_kg", "llm_kg"})
            self.assertEqual((tests[0]["system_a"], tests[0]["system_b"]), ("llm_kg", "metadata_kg"))
            self.assertIn("list_stats", results["llm_kg"])
            self.assertGreater(results["llm_kg"]["summary"]["all"]["coverage"], 0)

            audit = sample_explanations(files, root / "catalog.csv", root / "audit", n=10)
            with open(audit["A"], encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row["explanation"].startswith("(Movie ") for row in rows))


if __name__ == "__main__":
    unittest.main()

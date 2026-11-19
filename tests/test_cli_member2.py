import csv
import gzip
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def write_catalog(path):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "title", "overview", "genres", "keywords"))
        for movie_id, words in ((1, "cats adventure"), (2, "cats adventure"), (3, "motor racing")):
            writer.writerow((movie_id, f"Movie {movie_id}", words, '["Drama"]', "[]"))


def write_ratings(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        writer.writerows(rows)


class Member2CliTests(unittest.TestCase):
    def test_train_itemknn_and_evaluate_commands(self):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_catalog(directory / "catalog.csv")
            write_ratings(directory / "train.csv.gz", [
                (user, 1, 5, user) for user in range(1, 6)
            ] + [
                (user, 2, 5, user) for user in range(2, 6)
            ])
            write_ratings(directory / "test.csv.gz", [
                (1, 3, 5, 100),
            ])
            knn = subprocess.run(
                [
                    sys.executable, "-m", "movie_rec.cli", "train-itemknn",
                    "--catalog", str(directory / "catalog.csv"),
                    "--train", str(directory / "train.csv.gz"),
                    "--output", str(directory / "itemknn_model.npz"),
                    "--min-overlap", "5",
                ],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )
            self.assertIn("min_overlap=5", knn.stdout)
            tfidf = subprocess.run(
                [
                    sys.executable, "-m", "movie_rec.cli", "train-tfidf",
                    "--catalog", str(directory / "catalog.csv"),
                    "--output", str(directory / "tfidf_model.npz"),
                ],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )
            self.assertIn("Saved", tfidf.stdout)
            evaluated = subprocess.run(
                [
                    sys.executable, "-m", "movie_rec.cli", "evaluate",
                    "--directory", str(directory),
                    "--catalog", str(directory / "catalog.csv"),
                    "--tfidf-model", str(directory / "tfidf_model.npz"),
                    "--itemknn-model", str(directory / "itemknn_model.npz"),
                    "--k", "2",
                    "--output-dir", str(directory),
                ],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )
            payload = json.loads(evaluated.stdout)
            self.assertEqual(payload["k"], 2)
            self.assertTrue(payload["exclude_seed_only"])
            self.assertEqual(payload["models"]["tfidf"]["n_queries"], 1)
            self.assertEqual(payload["models"]["itemknn"]["n_queries"], 1)
            self.assertTrue((directory / "baseline_metrics.csv").exists())
            recommend = subprocess.run(
                [
                    sys.executable, "-m", "movie_rec.cli", "recommend", "1",
                    "--catalog", str(directory / "catalog.csv"),
                    "--model", str(directory / "itemknn_model.npz"),
                    "--k", "2",
                ],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )
            self.assertIn("movie_id,title,cosine_similarity", recommend.stdout.splitlines()[0])
            self.assertNotIn(",Movie 1,", recommend.stdout)


if __name__ == "__main__":
    unittest.main()

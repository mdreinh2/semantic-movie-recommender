import csv
import gc
import gzip
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from movie_rec.evaluate import evaluate_models, load_evaluation_protocol
from movie_rec.metrics import ranking_metrics


def write_catalog(path):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "title", "overview", "genres", "keywords"))
        for movie_id in (1, 2, 3, 4):
            writer.writerow((movie_id, f"Movie {movie_id}", "plot", "[]", "[]"))


def write_ratings(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        writer.writerows(rows)


class FixedRecommender:
    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def recommend(self, seed_movie_id, k=10, exclude=()):
        self.calls.append((seed_movie_id, k, tuple(exclude)))
        return [(movie_id, f"Movie {movie_id}", 1.0) for movie_id in self.ranking if movie_id != seed_movie_id][:k]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_catalog(self.root / "catalog.csv")
        # User 1 liked 1 in train and 3 in test. User 2 has no test positive, so no queries.
        write_ratings(self.root / "train.csv.gz", [
            (1, 1, 5.0, 1),
            (1, 2, 3.0, 2),
            (2, 1, 5.0, 1),
        ])
        write_ratings(self.root / "test.csv.gz", [
            (1, 3, 5.0, 10),
            (1, 4, 2.0, 11),
            (2, 3, 3.0, 10),
        ])
        (self.root / "manifest.json").write_text(json.dumps({
            "positive_rating_threshold": 4.0,
        }))

    def tearDown(self):
        gc.collect()

    def test_csv_protocol_matches_member1_views(self):
        protocol = load_evaluation_protocol(self.root)
        self.assertEqual(protocol["queries"], [(1, 1)])
        self.assertEqual(protocol["labels"][1], {3})
        self.assertTrue(protocol["exclude_seed_only"])
        self.assertNotIn(2, protocol["labels"])

    def test_sqlite_protocol_uses_existing_views(self):
        database = self.root / "experiment.sqlite"
        with sqlite3.connect(database) as db:
            db.executescript("""
                CREATE TABLE catalog (movie_id INTEGER PRIMARY KEY);
                CREATE TABLE ratings (
                    user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER, split TEXT
                );
                CREATE VIEW train_ratings AS
                    SELECT user_id, movie_id, rating, timestamp FROM ratings WHERE split = 'train';
                CREATE VIEW test_ratings AS
                    SELECT user_id, movie_id, rating, timestamp FROM ratings WHERE split = 'test';
                CREATE VIEW evaluation_labels AS
                    SELECT user_id, movie_id FROM test_ratings JOIN catalog USING (movie_id)
                    WHERE rating >= 4.0;
                CREATE VIEW seed_queries AS
                    SELECT r.user_id, r.movie_id AS seed_movie_id
                    FROM train_ratings r JOIN catalog USING (movie_id)
                    WHERE r.rating >= 4.0 AND EXISTS (
                        SELECT 1 FROM evaluation_labels l WHERE l.user_id = r.user_id
                    );
            """)
            db.executemany("INSERT INTO catalog VALUES (?)", [(1,), (2,), (3,), (4,)])
            db.executemany(
                "INSERT INTO ratings VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 1, 5.0, 1, "train"),
                    (1, 2, 3.0, 2, "train"),
                    (1, 3, 5.0, 10, "test"),
                    (8, 1, 5.0, 1, "train"),
                    (8, 4, 5.0, 10, "test"),
                ],
            )
        protocol = load_evaluation_protocol(self.root)
        self.assertEqual(protocol["queries"], [(1, 1), (8, 1)])
        self.assertEqual(protocol["labels"][8], {4})

    def test_same_protocol_for_two_models_and_seed_is_excluded(self):
        first = FixedRecommender([2, 3, 4])
        second = FixedRecommender([4, 2, 3])
        summaries, query_tables, protocol, paths = evaluate_models(
            {"alpha": first, "beta": second},
            self.root,
            k=2,
            output_dir=self.root / "out",
        )
        self.assertEqual(protocol["queries"], [(1, 1)])
        self.assertEqual(first.calls[0][:2], (1, 2))
        self.assertEqual(first.calls[0][2], ())
        expected_alpha = ranking_metrics([2, 3], {3}, 2)
        expected_beta = ranking_metrics([4, 2], {3}, 2)
        self.assertAlmostEqual(summaries["alpha"]["precision"], expected_alpha["precision"])
        self.assertAlmostEqual(summaries["beta"]["precision"], expected_beta["precision"])
        self.assertEqual(summaries["alpha"]["n_queries"], 1)
        self.assertEqual(query_tables["alpha"][0]["seed_movie_id"], 1)
        self.assertTrue((self.root / "out" / "baseline_metrics.csv").exists())
        self.assertTrue((self.root / "out" / "baseline_query_metrics.csv").exists())

    def test_limit_is_debug_only_and_positive(self):
        write_ratings(self.root / "train.csv.gz", [
            (1, 1, 5.0, 1),
            (1, 4, 5.0, 2),
        ])
        write_ratings(self.root / "test.csv.gz", [(1, 3, 5.0, 10)])
        protocol = load_evaluation_protocol(self.root, limit=1)
        self.assertEqual(len(protocol["queries"]), 1)
        with self.assertRaises(ValueError):
            load_evaluation_protocol(self.root, limit=0)


if __name__ == "__main__":
    unittest.main()

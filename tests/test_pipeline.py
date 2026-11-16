import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from movie_rec.handoff import export_handoff
from movie_rec.prepare import prepare
from movie_rec.tfidf import TfidfRecommender


def write_csv(path, header, records):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(records)


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.ml, self.tmdb, self.out = [root / name for name in ("ml", "tmdb", "out")]
        self.ml.mkdir()
        self.tmdb.mkdir()
        write_csv(self.ml / "links.csv", ("movieId", "tmdbId"),
                  [(i, 100 + i) for i in range(1, 26)] + [(99, 101)])
        write_csv(self.tmdb / "movies_metadata.csv",
                  ("id", "title", "overview", "genres", "release_date"),
                  [(i + 100, f"Movie {i}", "cat adventure" if i in (1, 24) else "motor racing",
                    "[]", "2020-01-01") for i in range(1, 26)] +
                  [("bad", "Invalid", "ignored", "[]", "")])
        write_csv(self.tmdb / "keywords.csv", ("id", "keywords"),
                  [(101, "[{'name': 'cat'}]")])
        write_csv(self.tmdb / "credits.csv", ("id", "cast", "crew"),
                  [(101, "[]", "[{'name': 'A Director', 'job': 'Director'}]")])
        write_csv(self.ml / "ratings.csv", ("userId", "movieId", "rating", "timestamp"),
                  [(1, i, 4, i // 3) for i in range(1, 26)] +
                  [(1, 99, 2, 0), (2, 1, 4, 1), (1, 3, 8, 100)])
        self.manifest = prepare(self.ml, self.tmdb, self.out)

    def test_aliases_chronology_and_train_only_popularity(self):
        counts = self.manifest["counts"]
        self.assertEqual(counts["train_ratings"], 20)
        self.assertEqual(counts["test_ratings"], 5)
        self.assertEqual(counts["duplicate_ratings"], 1)
        self.assertEqual(counts["invalid_ratings"], 1)
        with sqlite3.connect(self.out / "experiment.sqlite") as db:
            self.assertEqual(
                db.execute("SELECT movie_id FROM test_ratings ORDER BY timestamp, movie_id").fetchall(),
                [(i,) for i in range(21, 26)],
            )
            self.assertEqual(db.execute("SELECT train_count FROM catalog WHERE movie_id=24").fetchone(), (0,))
            self.assertEqual(db.execute("SELECT movie_id FROM movie_id_map WHERE source_movie_id=99").fetchone(), (1,))
            self.assertEqual(db.execute("SELECT directors FROM movies WHERE movie_id=1").fetchone(), ('["A Director"]',))

    def test_saved_model_preserves_ranking_and_text_vectors(self):
        model = TfidfRecommender(self.out / "catalog.csv")
        self.assertEqual(model.recommend(1)[0][0], 24)
        destination = self.out / "model.npz"
        model.save(destination)
        loaded = TfidfRecommender.load(destination)
        self.assertEqual(model.recommend(1), loaded.recommend(1))
        np.testing.assert_allclose(
            model.vectorizer.transform(["cat adventure"]).toarray(),
            loaded.vectorizer.transform(["cat adventure"]).toarray(),
        )
        self.assertFalse(any(movie_id in (1, 24) for movie_id, _, _ in loaded.recommend(1, exclude={24})))
        with self.assertRaises(ValueError):
            loaded.recommend(999)
        with self.assertRaises(ValueError):
            loaded.recommend(1, k=0)

    def test_exports_match_split_assignments(self):
        counts = export_handoff(self.out)
        self.assertEqual(counts["train.csv.gz"], 20)
        self.assertEqual(counts["test.csv.gz"], 5)
        with gzip.open(self.out / "test.csv.gz", "rt") as source:
            test_ids = [int(row["movie_id"]) for row in csv.DictReader(source)]
        self.assertEqual(test_ids, list(range(21, 26)))
        with (self.out / "catalog_ids.csv").open() as source:
            self.assertEqual(len(list(csv.DictReader(source))), 25)


if __name__ == "__main__":
    unittest.main()

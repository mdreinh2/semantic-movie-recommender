import csv
import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from movie_rec.itemknn import ItemKNNRecommender, load_recommender
from movie_rec.tfidf import TfidfRecommender


def write_catalog(path, movies):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "title", "overview", "genres", "keywords"))
        for movie_id, title in movies:
            writer.writerow((movie_id, title, f"{title} overview", "[]", "[]"))


def write_ratings(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        writer.writerows(rows)


class ItemKNNTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog.csv"
        self.train = self.root / "train.csv.gz"
        self.test = self.root / "test.csv.gz"
        write_catalog(self.catalog, ((1, "Alpha"), (2, "Beta"), (3, "Gamma"), (4, "Delta")))
        # Movies 1 and 2 share five raters; 1 and 3 share one rater; 4 is unused in train.
        write_ratings(self.train, [
            (user, 1, 5, user) for user in range(1, 6)
        ] + [
            (user, 2, 5, user) for user in range(1, 6)
        ] + [
            (1, 3, 5, 10),
        ])
        write_ratings(self.test, [
            (user, 1, 5, 100) for user in range(1, 6)
        ] + [
            (user, 3, 5, 100) for user in range(1, 6)
        ])

    def test_neighbors_follow_training_co_ratings(self):
        model = ItemKNNRecommender(self.catalog, self.train, min_overlap=5)
        ranked = model.recommend(1, k=3)
        self.assertEqual(ranked[0][0], 2)
        self.assertGreater(ranked[0][2], 0.0)
        by_id = {movie_id: score for movie_id, _, score in ranked}
        self.assertEqual(by_id[3], 0.0)
        self.assertNotIn(1, by_id)

    def test_min_overlap_is_configurable(self):
        strict = ItemKNNRecommender(self.catalog, self.train, min_overlap=5)
        loose = ItemKNNRecommender(self.catalog, self.train, min_overlap=1)
        self.assertEqual(strict.recommend(1, k=3)[0][0], 2)
        self.assertGreater(loose.recommend(1, k=3)[1][2], 0.0)

    def test_does_not_read_test_ratings_while_fitting(self):
        opened = []
        original_open = gzip.open

        def tracking_open(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_open(path, *args, **kwargs)

        with patch("movie_rec.itemknn.gzip.open", tracking_open):
            ItemKNNRecommender(self.catalog, self.train, min_overlap=5)
        self.assertEqual(opened, ["train.csv.gz"])
        self.assertNotIn("test.csv.gz", opened)

        train_only = ItemKNNRecommender(self.catalog, self.train, min_overlap=5)
        leaked = ItemKNNRecommender(self.catalog, self.test, min_overlap=5)
        self.assertEqual(train_only.recommend(1, k=1)[0][0], 2)
        self.assertEqual(leaked.recommend(1, k=1)[0][0], 3)

    def test_save_and_load_preserve_ranking(self):
        model = ItemKNNRecommender(self.catalog, self.train, min_overlap=5)
        destination = self.root / "itemknn_model.npz"
        model.save(destination)
        loaded = ItemKNNRecommender.load(destination)
        self.assertEqual(model.recommend(1, k=3), loaded.recommend(1, k=3))
        self.assertEqual(loaded.min_overlap, 5)
        self.assertEqual(load_recommender(destination).recommend(1, k=1)[0][0], 2)

    def test_load_recommender_keeps_tfidf_artifacts(self):
        tfidf = TfidfRecommender(self.catalog)
        destination = self.root / "tfidf_model.npz"
        tfidf.save(destination)
        loaded = load_recommender(destination)
        self.assertEqual(tfidf.recommend(1, k=2), loaded.recommend(1, k=2))

    def test_exclude_and_validation(self):
        model = ItemKNNRecommender(self.catalog, self.train, min_overlap=1)
        ranked = model.recommend(1, k=3, exclude={2})
        self.assertFalse(any(movie_id in (1, 2) for movie_id, _, _ in ranked))
        with self.assertRaises(ValueError):
            model.recommend(99)
        with self.assertRaises(ValueError):
            model.recommend(1, k=0)
        with self.assertRaises(ValueError):
            ItemKNNRecommender(self.catalog, self.train, min_overlap=0)


if __name__ == "__main__":
    unittest.main()

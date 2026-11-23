"""Item-based k-nearest-neighbor baseline fitted on training ratings only."""

from __future__ import annotations

import csv
import gzip
import hashlib
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def file_sha256(path):
    with Path(path).open("rb") as source:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(source, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _read_catalog(catalog_path):
    with Path(catalog_path).open(encoding="utf-8", newline="") as source:
        movies = sorted(csv.DictReader(source), key=lambda row: int(row["movie_id"]))
    if not movies:
        raise ValueError("The catalog is empty.")
    movie_ids = np.array([int(movie["movie_id"]) for movie in movies], dtype=np.int64)
    titles = [movie["title"] for movie in movies]
    index = {int(movie_id): i for i, movie_id in enumerate(movie_ids)}
    if len(index) != len(movies):
        raise ValueError("The catalog contains duplicate movie IDs.")
    return movie_ids, titles, index


def _iter_train_ratings(train_path, catalog_index):
    with _open_text(train_path) as source:
        for row in csv.DictReader(source):
            movie_id = int(row["movie_id"])
            if movie_id not in catalog_index:
                continue
            yield int(row["user_id"]), movie_id, float(row["rating"])


class ItemKNNRecommender:
    """Rank catalog movies by cosine similarity of training rating vectors."""

    def __init__(self, catalog_path, train_path, min_overlap=5):
        if min_overlap < 1:
            raise ValueError("min_overlap must be at least 1.")
        catalog_path = Path(catalog_path)
        train_path = Path(train_path)
        self.min_overlap = int(min_overlap)
        self.catalog_sha256 = file_sha256(catalog_path)
        self.train_sha256 = file_sha256(train_path)
        self.movie_ids, self.titles, self.index = _read_catalog(catalog_path)
        self.similarity = self._fit(_iter_train_ratings(train_path, self.index))

    def _fit(self, ratings):
        users = {}
        rows, cols, values = [], [], []
        for user_id, movie_id, rating in ratings:
            user_index = users.setdefault(user_id, len(users))
            rows.append(user_index)
            cols.append(self.index[movie_id])
            values.append(rating)
        n_items = len(self.movie_ids)
        if not values:
            raise ValueError("No training ratings overlap the shared catalog.")
        matrix = coo_matrix((values, (rows, cols)), shape=(len(users), n_items)).tocsr()
        support = matrix.copy()
        support.data = np.ones(support.nnz, dtype=np.float64)
        co_counts = (support.T @ support).toarray()
        similarity = cosine_similarity(matrix.T, dense_output=True)
        similarity[co_counts < self.min_overlap] = 0.0
        np.fill_diagonal(similarity, 0.0)
        return similarity.astype(np.float32, copy=False)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            pending = Path(temporary) / "model.npz"
            matrix = csr_matrix(self.similarity)
            np.savez_compressed(
                pending,
                format_version=np.array(1),
                model_type=np.array("itemknn"),
                movie_ids=self.movie_ids,
                titles=np.asarray(self.titles),
                data=matrix.data,
                indices=matrix.indices,
                indptr=matrix.indptr,
                shape=np.asarray(matrix.shape),
                min_overlap=np.array(self.min_overlap),
                catalog_sha256=np.array(self.catalog_sha256),
                train_sha256=np.array(self.train_sha256),
            )
            pending.replace(path)

    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["format_version"]) != 1:
                raise ValueError("Unsupported Item-KNN model format.")
            if str(saved["model_type"]) != "itemknn":
                raise ValueError("The saved file is not an Item-KNN model.")
            model.movie_ids = saved["movie_ids"]
            model.titles = saved["titles"].tolist()
            model.index = {int(movie_id): i for i, movie_id in enumerate(model.movie_ids)}
            model.similarity = csr_matrix(
                (saved["data"], saved["indices"], saved["indptr"]),
                shape=tuple(saved["shape"]),
            ).toarray()
            model.min_overlap = int(saved["min_overlap"])
            model.catalog_sha256 = str(saved["catalog_sha256"])
            model.train_sha256 = str(saved["train_sha256"])
        if model.similarity.shape[0] != len(model.movie_ids):
            raise ValueError("Model similarities and movie IDs are inconsistent.")
        return model

    def recommend(self, seed_movie_id, k=10, exclude=()):
        if k < 1:
            raise ValueError("k must be positive.")
        if seed_movie_id not in self.index:
            raise ValueError(f"MovieLens ID {seed_movie_id} is outside the shared catalog.")
        seed_idx = self.index[seed_movie_id]
        scores = np.asarray(self.similarity[seed_idx], dtype=np.float64).ravel()
        eligible = np.ones(len(self.movie_ids), dtype=bool)
        eligible[seed_idx] = False
        for movie_id in exclude:
            if movie_id in self.index:
                eligible[self.index[movie_id]] = False
        ranked = np.flatnonzero(eligible)
        ranked = ranked[np.argsort(-scores[ranked], kind="stable")[:k]]
        return [
            (int(self.movie_ids[i]), self.titles[i], float(scores[i]))
            for i in ranked
        ]


def load_recommender(path):
    """Load a TF-IDF or Item-KNN artifact from a compressed .npz file."""
    with np.load(path, allow_pickle=False) as saved:
        model_type = str(saved["model_type"]) if "model_type" in saved.files else "tfidf"
    if model_type == "itemknn":
        return ItemKNNRecommender.load(path)
    if model_type != "tfidf":
        raise ValueError(f"Unsupported recommender type {model_type!r}.")
    from .tfidf import TfidfRecommender
    return TfidfRecommender.load(path)

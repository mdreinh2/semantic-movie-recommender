"""Seed-to-movie ranking over the shared catalog."""

import csv
import json
import hashlib
import tempfile
from pathlib import Path

from scipy.sparse import csr_matrix

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRecommender:
    def __init__(self, catalog_path):
        self.catalog_sha256 = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
        with open(catalog_path, encoding="utf-8", newline="") as source:
            movies = sorted(csv.DictReader(source), key=lambda row: int(row["movie_id"]))
        if not movies:
            raise ValueError("The catalog is empty.")
        self.movie_ids = np.array([int(movie["movie_id"]) for movie in movies])
        self.titles = [movie["title"] for movie in movies]
        self.index = {movie_id: i for i, movie_id in enumerate(self.movie_ids)}
        if len(self.index) != len(movies):
            raise ValueError("The catalog contains duplicate movie IDs.")
        documents = [
            " ".join([movie["overview"], *json.loads(movie["genres"]), *json.loads(movie["keywords"])])
            for movie in movies
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", norm="l2")
        if not any(self.vectorizer.build_analyzer()(document) for document in documents):
            raise ValueError("The catalog has no usable overview, genre, or keyword tokens.")
        self.vectors = self.vectorizer.fit_transform(documents)

    def save(self, path):
        """Persist vectors, vocabulary, IDF weights and movie order without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            pending = Path(temporary) / "model.npz"
            np.savez_compressed(
                pending,
                format_version=np.array(1),
                movie_ids=self.movie_ids,
                titles=np.asarray(self.titles),
                data=self.vectors.data,
                indices=self.vectors.indices,
                indptr=self.vectors.indptr,
                shape=np.asarray(self.vectors.shape),
                terms=self.vectorizer.get_feature_names_out().astype(str),
                idf=self.vectorizer.idf_,
                catalog_sha256=np.array(self.catalog_sha256),
            )
            pending.replace(path)

    @classmethod
    def load(cls, path):
        """Load a fitted catalog model for repeated recommendations."""
        model = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["format_version"]) != 1:
                raise ValueError("Unsupported TF-IDF model format.")
            model.movie_ids = saved["movie_ids"]
            model.titles = saved["titles"].tolist()
            model.index = {int(movie): i for i, movie in enumerate(model.movie_ids)}
            model.vectors = csr_matrix(
                (saved["data"], saved["indices"], saved["indptr"]),
                shape=tuple(saved["shape"]),
            )
            model.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), stop_words="english", norm="l2",
                vocabulary={term: i for i, term in enumerate(saved["terms"].tolist())},
            )
            model.vectorizer.idf_ = saved["idf"]
            model.catalog_sha256 = str(saved["catalog_sha256"])
        if model.vectors.shape[0] != len(model.movie_ids) or len(model.index) != len(model.movie_ids):
            raise ValueError("Model vectors and movie IDs are inconsistent.")
        return model

    def recommend(self, seed_movie_id, k=10, exclude=()):
        """Return (movie ID, title, cosine score) tuples ordered by relevance."""
        if k < 1:
            raise ValueError("k must be positive.")
        if seed_movie_id not in self.index:
            raise ValueError(f"MovieLens ID {seed_movie_id} is outside the shared catalog.")
        seed_idx = self.index[seed_movie_id]
        scores = (self.vectors @ self.vectors[seed_idx].T).toarray().ravel()
        eligible = np.ones(len(self.movie_ids), dtype=bool)
        eligible[seed_idx] = False
        for movie_id in exclude:
            if movie_id in self.index:
                eligible[self.index[movie_id]] = False
        ranked = np.flatnonzero(eligible)
        # Inputs are already sorted by movie ID, so a stable sort resolves ties.
        ranked = ranked[np.argsort(-scores[ranked], kind="stable")[:k]]
        return [
            (int(self.movie_ids[i]), self.titles[i], float(scores[i]))
            for i in ranked
        ]

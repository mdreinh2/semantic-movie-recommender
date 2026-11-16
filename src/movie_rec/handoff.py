"""Export the shared data and verify the packaged files."""

import csv
import gzip
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from .prepare import MOVIE_COLUMNS


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_csv(path, cursor, header):
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        while batch := cursor.fetchmany(50_000):
            writer.writerows(batch)
            count += len(batch)
    return count


def export_handoff(directory):
    directory = Path(directory)
    database = directory / "experiment.sqlite"
    manifest = json.loads((directory / "manifest.json").read_text())
    exports = {
        "train.csv.gz": (
            "SELECT * FROM train_ratings ORDER BY user_id, movie_id",
            ("user_id", "movie_id", "rating", "timestamp"),
        ),
        "test.csv.gz": (
            "SELECT * FROM test_ratings ORDER BY user_id, movie_id",
            ("user_id", "movie_id", "rating", "timestamp"),
        ),
        "movies_clean.csv.gz": (
            "SELECT * FROM movies ORDER BY movie_id", MOVIE_COLUMNS,
        ),
        "movie_id_map.csv": (
            "SELECT * FROM movie_id_map ORDER BY source_movie_id",
            ("source_movie_id", "movie_id"),
        ),
        "catalog_ids.csv": (
            "SELECT movie_id, tmdb_id FROM catalog JOIN movies USING(movie_id) ORDER BY movie_id",
            ("movie_id", "tmdb_id"),
        ),
        "split_manifest.csv.gz": (
            """SELECT user_id, COUNT(*) AS n,
                SUM(split='train'), SUM(split='test'),
                MAX(CASE WHEN split='train' THEN timestamp END),
                MIN(CASE WHEN split='test' THEN timestamp END)
            FROM ratings GROUP BY user_id ORDER BY user_id""",
            ("user_id", "n_ratings", "n_train", "n_test", "latest_train_timestamp", "earliest_test_timestamp"),
        ),
    }
    counts = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        catalog_ids = [
            (int(row["movie_id"]), int(row["tmdb_id"]))
            for row in csv.DictReader((directory / "catalog.csv").read_text(encoding="utf-8").splitlines())
        ]
        expected_ids = db.execute(
            "SELECT movie_id, tmdb_id FROM catalog JOIN movies USING(movie_id) ORDER BY movie_id"
        ).fetchall()
        if catalog_ids != expected_ids:
            raise ValueError("Catalog and database movie IDs differ. Run prepare again.")
        with tempfile.TemporaryDirectory(dir=directory, prefix=".export-") as temporary:
            staging = Path(temporary)
            for name, (query, header) in exports.items():
                counts[name] = write_csv(staging / name, db.execute(query), header)
                print(f"Exported {name}: {counts[name]:,} rows", flush=True)
            if counts["train.csv.gz"] != manifest["counts"]["train_ratings"]:
                raise ValueError("Train count differs from the preparation manifest.")
            if counts["test.csv.gz"] != manifest["counts"]["test_ratings"]:
                raise ValueError("Test count differs from the preparation manifest.")
            for name in exports:
                (staging / name).replace(directory / name)
    return counts


def seal_handoff(directory):
    """Record the current payload after fitting and ranking have completed."""
    directory = Path(directory)
    names = (
        "catalog.csv", "catalog_ids.csv", "manifest.json", "movies_clean.csv.gz",
        "movie_id_map.csv", "split_manifest.csv.gz", "train.csv.gz", "test.csv.gz",
        "tfidf_model.npz", "tfidf_catalog_top10.csv",
    )
    manifest = {name: {"sha256": sha256(directory / name),
                       "bytes": (directory / name).stat().st_size} for name in names}
    temporary = directory / "checksums.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "checksums.json")


def verify_handoff(directory):
    from .tfidf import TfidfRecommender

    directory = Path(directory)
    checksums = json.loads((directory / "checksums.json").read_text())
    for name, expected in checksums.items():
        if sha256(directory / name) != expected["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}.")
    model = TfidfRecommender.load(directory / "tfidf_model.npz")
    if model.catalog_sha256 != sha256(directory / "catalog.csv"):
        raise ValueError("The TF-IDF model was fitted on a different catalog.")
    return {
        "verified_files": len(checksums),
        "catalog_movies": len(model.movie_ids),
        "tfidf_features": model.vectors.shape[1],
    }

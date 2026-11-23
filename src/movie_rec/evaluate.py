"""Shared evaluation protocol for catalog seed-to-item recommenders.

The protocol is the one already encoded by Member 1:

* Candidate pool = shared catalog
* Seed queries = catalog movies a user rated >= 4.0 in TRAINING, for users who
  have at least one catalog test rating >= 4.0 (`seed_queries`)
* Ground truth = catalog movies those users rated >= 4.0 in TEST
  (`evaluation_labels`)
* The seed movie is always excluded (Member 1 `recommend()` / VALIDATION.md)
* Previously rated training movies are not additionally filtered: the PPT and
  Member 1 docs require the same catalog, seeds, labels, and Top-K, and they
  only document excluding the seed
* Metrics are macro-averaged over evaluation queries
"""

from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from .metrics import ranking_metrics

POSITIVE_RATING = 4.0


def _open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _catalog_ids(directory):
    path = Path(directory) / "catalog.csv"
    with path.open(encoding="utf-8", newline="") as source:
        return {int(row["movie_id"]) for row in csv.DictReader(source)}


def _positive_threshold(directory):
    manifest = Path(directory) / "manifest.json"
    if not manifest.exists():
        return POSITIVE_RATING
    payload = json.loads(manifest.read_text())
    return float(payload.get("positive_rating_threshold", POSITIVE_RATING))


def _from_sqlite(directory, catalog_ids, threshold):
    database = Path(directory) / "experiment.sqlite"
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        queries = db.execute(
            "SELECT user_id, seed_movie_id FROM seed_queries ORDER BY user_id, seed_movie_id"
        ).fetchall()
        labels = defaultdict(set)
        for user_id, movie_id in db.execute(
            "SELECT user_id, movie_id FROM evaluation_labels"
        ):
            if movie_id in catalog_ids:
                labels[int(user_id)].add(int(movie_id))
    finally:
        db.close()
    return [(int(user), int(seed)) for user, seed in queries], dict(labels), threshold



def _from_csvs(directory, catalog_ids, threshold):
    train = Path(directory) / "train.csv.gz"
    test = Path(directory) / "test.csv.gz"
    labels = defaultdict(set)
    with _open_text(test) as source:
        for row in csv.DictReader(source):
            movie_id = int(row["movie_id"])
            if movie_id in catalog_ids and float(row["rating"]) >= threshold:
                labels[int(row["user_id"])].add(movie_id)
    queries = []
    with _open_text(train) as source:
        for row in csv.DictReader(source):
            user_id = int(row["user_id"])
            movie_id = int(row["movie_id"])
            if (
                movie_id in catalog_ids
                and float(row["rating"]) >= threshold
                and user_id in labels
            ):
                queries.append((user_id, movie_id))
    queries.sort()
    return queries, dict(labels), threshold


def load_evaluation_protocol(directory, limit=None):
    """Load seed queries and test labels from Member 1 outputs."""
    directory = Path(directory)
    catalog_ids = _catalog_ids(directory)
    if not catalog_ids:
        raise ValueError("The catalog is empty.")
    threshold = _positive_threshold(directory)
    if (directory / "experiment.sqlite").exists():
        queries, labels, threshold = _from_sqlite(directory, catalog_ids, threshold)
    else:
        queries, labels, threshold = _from_csvs(directory, catalog_ids, threshold)
    queries = [
        (user_id, seed_id) for user_id, seed_id in queries
        if seed_id in catalog_ids and labels.get(user_id)
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        queries = queries[:limit]
    if not queries:
        raise ValueError("No evaluation queries were found.")
    return {
        "queries": queries,
        "labels": labels,
        "catalog_ids": catalog_ids,
        "positive_rating_threshold": threshold,
        "exclude_seed_only": True,
    }


def evaluate_recommender(model, protocol, k=10):
    """Score one recommender under the shared protocol."""
    if k < 1:
        raise ValueError("k must be positive.")
    rankings = {}
    query_rows = []
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
    for user_id, seed_id in protocol["queries"]:
        if seed_id not in rankings:
            rankings[seed_id] = [movie_id for movie_id, _, _ in model.recommend(seed_id, k)]
        relevant = protocol["labels"][user_id]
        metrics = ranking_metrics(rankings[seed_id], relevant, k)
        query_rows.append({
            "user_id": user_id,
            "seed_movie_id": seed_id,
            **metrics,
        })
        for name, value in metrics.items():
            totals[name] += value
    n_queries = len(query_rows)
    summary = {name: totals[name] / n_queries for name in totals}
    summary.update({"k": k, "n_queries": n_queries})
    return summary, query_rows


def write_evaluation_outputs(output_dir, summaries, query_tables, k):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_metrics.csv"
    json_path = output_dir / "baseline_metrics.json"
    queries_path = output_dir / "baseline_query_metrics.csv"
    fieldnames = ("model", "k", "precision", "recall", "ndcg", "hit_rate", "n_queries")
    with metrics_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, summary in summaries.items():
            writer.writerow({
                "model": name,
                "k": k,
                "precision": summary["precision"],
                "recall": summary["recall"],
                "ndcg": summary["ndcg"],
                "hit_rate": summary["hit_rate"],
                "n_queries": summary["n_queries"],
            })
    json_path.write_text(json.dumps({
        "k": k,
        "exclude_seed_only": True,
        "models": summaries,
    }, indent=2) + "\n")
    with queries_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("model", "user_id", "seed_movie_id", "precision", "recall", "ndcg", "hit_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        for name, rows in query_tables.items():
            for row in rows:
                writer.writerow({"model": name, **row})
    return metrics_path, json_path, queries_path


def evaluate_models(models, directory, k=10, limit=None, output_dir=None):
    """Evaluate named recommenders and optionally write comparison files."""
    protocol = load_evaluation_protocol(directory, limit=limit)
    summaries = {}
    query_tables = {}
    for name, model in models.items():
        summaries[name], query_tables[name] = evaluate_recommender(model, protocol, k)
    paths = None
    if output_dir is not None:
        paths = write_evaluation_outputs(output_dir, summaries, query_tables, k)
    return summaries, query_tables, protocol, paths

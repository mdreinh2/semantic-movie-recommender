"""Command-line entry points for the Member 1 data and TF-IDF pipeline."""

import argparse
import csv
import json
import sqlite3
import sys
from contextlib import closing, nullcontext
from pathlib import Path

from .prepare import prepare


def main():
    command = argparse.ArgumentParser(prog="movie-rec")
    subcommands = command.add_subparsers(dest="command", required=True)
    prep = subcommands.add_parser("prepare", help="clean, link, split, and sample")
    prep.add_argument("--movielens", type=Path, default=Path("data/raw/ml-20m"))
    prep.add_argument("--tmdb", type=Path, default=Path("data/raw/tmdb"))
    prep.add_argument("--output", type=Path, default=Path("data/processed"))
    prep.add_argument("--catalog-size", type=int, default=5000)
    prep.add_argument("--seed", type=int, default=42)

    export = subcommands.add_parser("export", help="export shared split and metadata files")
    export.add_argument("--directory", type=Path, default=Path("data/processed"))

    fit = subcommands.add_parser("train-tfidf", help="fit and save the TF-IDF model")
    fit.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    fit.add_argument("--output", type=Path, default=Path("data/processed/tfidf_model.npz"))

    rank = subcommands.add_parser("recommend", help="rank catalog candidates for a seed")
    rank.add_argument("movie_id", type=int, help="canonical MovieLens ID from catalog.csv")
    rank.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    rank.add_argument("--model", type=Path, help="load a saved model instead of refitting")
    rank.add_argument("--k", type=int, default=10)

    batch = subcommands.add_parser("rank-catalog", help="write seed movie Top-K rankings")
    batch.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    batch.add_argument("--database", type=Path, default=Path("data/processed/experiment.sqlite"))
    batch.add_argument("--model", type=Path)
    batch.add_argument("--all-catalog", action="store_true", help="rank all catalog seeds without reading ratings")
    batch.add_argument("--output", type=Path, help="write a UTF-8 CSV, defaults to stdout")
    batch.add_argument("--k", type=int, default=10)

    check = subcommands.add_parser("verify", help="check exported data and fitted model integrity")
    check.add_argument("--directory", type=Path, default=Path("data/processed"))

    args = command.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.movielens, args.tmdb, args.output, args.catalog_size, args.seed)
            print(json.dumps(result, indent=2))
            return
        if args.command == "export":
            from .handoff import export_handoff
            export_handoff(args.directory)
            return
        if args.command == "verify":
            from .handoff import verify_handoff
            print(json.dumps(verify_handoff(args.directory), indent=2))
            return

        from .tfidf import TfidfRecommender

        if getattr(args, "model", None):
            from .handoff import sha256
            model = TfidfRecommender.load(args.model)
            if model.catalog_sha256 != sha256(args.catalog):
                raise ValueError("The saved model and requested catalog differ.")
        else:
            model = TfidfRecommender(args.catalog)

        if args.command == "train-tfidf":
            model.save(args.output)
            print(f"Saved {len(model.movie_ids):,} movie vectors to {args.output}")
            return
        if args.k < 1:
            raise ValueError("k must be positive.")
        if args.command == "recommend":
            results = model.recommend(args.movie_id, args.k)
            writer = csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(("movie_id", "title", "cosine_similarity"))
            writer.writerows(results)
            return
        if args.all_catalog:
            seed_ids = model.movie_ids.tolist()
        else:
            with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                database_ids = [row[0] for row in db.execute("SELECT movie_id FROM catalog ORDER BY movie_id")]
                if database_ids != model.movie_ids.tolist():
                    raise ValueError("The database and model catalogs differ.")
                seed_ids = [
                    row[0] for row in db.execute(
                        "SELECT DISTINCT seed_movie_id FROM seed_queries ORDER BY seed_movie_id"
                    )
                ]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        target = args.output.open("w", encoding="utf-8", newline="") if args.output else nullcontext(sys.stdout)
        with target as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("seed_movie_id", "rank", "movie_id", "cosine_similarity"))
            for seed in seed_ids:
                for position, (movie_id, _, score) in enumerate(model.recommend(seed, args.k), 1):
                    writer.writerow((seed, position, movie_id, score))
    except (OSError, ValueError, sqlite3.Error) as error:
        command.error(str(error))


if __name__ == "__main__":
    main()

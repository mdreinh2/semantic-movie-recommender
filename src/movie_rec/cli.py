"""Command-line entry points for data preparation, baselines, evaluation, and extraction."""

import argparse
import csv
import json
import sqlite3
import sys
from contextlib import closing, nullcontext
from pathlib import Path

from .prepare import prepare


def _add_extraction_arguments(parser):
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    parser.add_argument("--prompt", type=Path, default=Path("prompts/extract_triples.txt"))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/member3"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180)


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

    knn = subcommands.add_parser("train-itemknn", help="fit Item-KNN on training ratings")
    knn.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    knn.add_argument("--train", type=Path, default=Path("data/processed/train.csv.gz"))
    knn.add_argument("--output", type=Path, default=Path("data/processed/itemknn_model.npz"))
    knn.add_argument("--min-overlap", type=int, default=5)

    ev = subcommands.add_parser("evaluate", help="score saved recommenders on the shared protocol")
    ev.add_argument("--directory", type=Path, default=Path("data/processed"))
    ev.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    ev.add_argument("--tfidf-model", type=Path, default=Path("data/processed/tfidf_model.npz"))
    ev.add_argument("--itemknn-model", type=Path, default=Path("data/processed/itemknn_model.npz"))
    ev.add_argument("--models", default="tfidf,itemknn", help="comma-separated names: tfidf, itemknn")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--limit", type=int, help="optional debug cap on evaluation queries")
    ev.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    pilot = subcommands.add_parser(
        "extract-pilot",
        help="extract triples for a fixed pilot sample with local Qwen",
    )
    _add_extraction_arguments(pilot)
    pilot.add_argument("--n", type=int, default=50, help="number of movies in the saved pilot sample")
    pilot.add_argument("--seed", type=int, default=42, help="sample seed used only when creating the pilot ID file")

    catalog_extract = subcommands.add_parser(
        "extract-catalog",
        help="resume full-catalog triple extraction for Member 4",
    )
    _add_extraction_arguments(catalog_extract)
    catalog_extract.add_argument(
        "--retry-failed", action="store_true",
        help="run movies whose last checkpoint status is extraction_failed",
    )

    final = subcommands.add_parser(
        "final-evaluate",
        help="Member 5: compare all systems at K=5/10/20 with significance and cold-start results",
    )
    final.add_argument(
        "--system", action="append", default=[], metavar="NAME=PATH",
        help="a saved model (.npz) or Top-K ranking file (.csv/.csv.gz); repeat per system",
    )
    final.add_argument("--directory", type=Path, default=Path("data/processed"))
    final.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    final.add_argument("--ks", default="5,10,20", help="comma-separated cutoffs")
    final.add_argument("--cold-start-max", type=int, default=10,
                       help="cold-start seeds have fewer than this many training ratings")
    final.add_argument("--target", help="system tested against every other one (default: llm_kg if given)")
    final.add_argument("--limit", type=int, help="optional debug cap on evaluation queries")
    final.add_argument("--output-dir", type=Path, default=Path("data/processed/member5"))
    final.add_argument("--write-query-metrics", action="store_true",
                       help="also write per-query NDCG@10 for every system")

    sample_t = subcommands.add_parser("audit-sample-triples", help="Member 5: blank triple review sheets")
    sample_t.add_argument("--triples", type=Path, default=Path("data/processed/member3/semantic_triples.csv"))
    sample_t.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    sample_t.add_argument("--n", type=int, default=200)
    sample_t.add_argument("--seed", type=int, default=42)
    sample_t.add_argument("--output-dir", type=Path, default=Path("data/processed/member5/audit"))

    score_t = subcommands.add_parser("audit-score-triples", help="Member 5: score two triple reviews")
    score_t.add_argument("--review-a", type=Path, default=Path("data/processed/member5/audit/triple_review_A.csv"))
    score_t.add_argument("--review-b", type=Path, default=Path("data/processed/member5/audit/triple_review_B.csv"))
    score_t.add_argument("--status", type=Path, default=Path("data/processed/member3/extraction_status.csv"))
    score_t.add_argument("--triples", type=Path, default=Path("data/processed/member3/semantic_triples.csv"))
    score_t.add_argument("--output-dir", type=Path, default=Path("data/processed/member5/audit"))

    sample_e = subcommands.add_parser(
        "audit-sample-explanations", help="Member 5: blind explanation review sheets",
    )
    sample_e.add_argument("--system", action="append", default=[], metavar="NAME=PATH",
                          help="KG ranking file with an explanation column; give two")
    sample_e.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    sample_e.add_argument("--n", type=int, default=30)
    sample_e.add_argument("--seed", type=int, default=42)
    sample_e.add_argument("--output-dir", type=Path, default=Path("data/processed/member5/audit"))

    score_e = subcommands.add_parser("audit-score-explanations", help="Member 5: score two explanation reviews")
    score_e.add_argument("--review-a", type=Path, default=Path("data/processed/member5/audit/explanation_review_A.csv"))
    score_e.add_argument("--review-b", type=Path, default=Path("data/processed/member5/audit/explanation_review_B.csv"))
    score_e.add_argument("--key", type=Path, default=Path("data/processed/member5/audit/explanation_audit_key.csv"))
    score_e.add_argument("--output-dir", type=Path, default=Path("data/processed/member5/audit"))

    args = command.parse_args()
    try:
        if args.command == "final-evaluate":
            from .final_eval import parse_system_specs, run_final_evaluation
            ks = tuple(int(value) for value in args.ks.split(",") if value.strip())
            _, tests, hypotheses, paths = run_final_evaluation(
                parse_system_specs(args.system), args.directory, args.output_dir, ks,
                args.limit, args.cold_start_max, args.target, args.catalog,
                args.write_query_metrics,
            )
            print(paths["report_md"].read_text(encoding="utf-8"))
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return
        if args.command == "audit-sample-triples":
            from .audit import sample_triples
            print(json.dumps(sample_triples(
                args.triples, args.catalog, args.output_dir, args.n, args.seed,
            ), indent=2))
            return
        if args.command == "audit-score-triples":
            from .audit import score_triples
            status = args.status if args.status.exists() else None
            triples = args.triples if args.triples.exists() else None
            print(json.dumps(score_triples(
                args.review_a, args.review_b, args.output_dir, status, triples,
            ), indent=2))
            return
        if args.command == "audit-sample-explanations":
            from .audit import sample_explanations
            from .final_eval import parse_system_specs
            print(json.dumps(sample_explanations(
                parse_system_specs(args.system), args.catalog, args.output_dir, args.n, args.seed,
            ), indent=2))
            return
        if args.command == "audit-score-explanations":
            from .audit import score_explanations
            print(json.dumps(score_explanations(
                args.review_a, args.review_b, args.key, args.output_dir,
            ), indent=2))
            return
        if args.command == "extract-pilot":
            from .extract import run_pilot
            result = run_pilot(
                args.catalog, args.prompt, args.output_dir, args.model, args.n, args.seed,
                args.ollama_url, args.timeout,
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "extract-catalog":
            from .extract import run_catalog
            result = run_catalog(
                args.catalog, args.prompt, args.output_dir, args.model,
                args.ollama_url, args.timeout, args.retry_failed,
            )
            print(json.dumps(result, indent=2))
            return
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
        if args.command == "train-itemknn":
            from .itemknn import ItemKNNRecommender
            model = ItemKNNRecommender(args.catalog, args.train, args.min_overlap)
            model.save(args.output)
            print(
                f"Saved Item-KNN for {len(model.movie_ids):,} movies "
                f"(min_overlap={model.min_overlap}) to {args.output}"
            )
            return
        if args.command == "evaluate":
            from .evaluate import evaluate_models
            from .itemknn import file_sha256, load_recommender
            requested = [name.strip() for name in args.models.split(",") if name.strip()]
            paths = {"tfidf": args.tfidf_model, "itemknn": args.itemknn_model}
            models = {}
            catalog_hash = file_sha256(args.catalog)
            for name in requested:
                if name not in paths:
                    raise ValueError(f"Unknown model {name!r}. Use tfidf or itemknn.")
                model = load_recommender(paths[name])
                if model.catalog_sha256 != catalog_hash:
                    raise ValueError(f"The {name} model and requested catalog differ.")
                models[name] = model
            summaries, _, _, output_paths = evaluate_models(
                models, args.directory, args.k, args.limit, args.output_dir,
            )
            n_queries = next(iter(summaries.values()))["n_queries"] if summaries else 0
            print(json.dumps({
                "k": args.k,
                "n_queries": n_queries,
                "exclude_seed_only": True,
                "models": summaries,
                "outputs": [str(path) for path in output_paths],
            }, indent=2))
            return

        from .tfidf import TfidfRecommender
        from .itemknn import load_recommender

        if getattr(args, "model", None):
            from .itemknn import file_sha256
            model = load_recommender(args.model)
            if model.catalog_sha256 != file_sha256(args.catalog):
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

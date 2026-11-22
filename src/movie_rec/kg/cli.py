import argparse
import csv
import json
import os
import sys
from pathlib import Path
from .graph_data import build_graph
from .memory_store import MemoryGraphStore
from .ranking import KnowledgeGraphRecommender
MODES={"metadata":("metadata",), "enhanced":("metadata", "llm")}
def add_graph_arguments(parser):
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    parser.add_argument("--triples", type=Path, help="Member 3 triples as CSV or JSONL")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--cast-limit", type=int, default=5)
def add_connection_arguments(parser):
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
def open_store(args, graph):
    if args.backend=="memory":
        return MemoryGraphStore(graph)
    from .neo4j_store import Neo4jGraphStore
    return Neo4jGraphStore(args.uri, args.user, args.password, args.database)
def main():
    command=argparse.ArgumentParser(prog="movie-kg")
    subcommands=command.add_subparsers(dest="command", required=True)
    load=subcommands.add_parser("load", help="load metadata and LLM triples into Neo4j")
    add_graph_arguments(load)
    add_connection_arguments(load)
    load.add_argument("--reset", action="store_true")
    recommend=subcommands.add_parser("recommend", help="Top-K recommendations with explanation paths")
    add_graph_arguments(recommend)
    add_connection_arguments(recommend)
    recommend.add_argument("movie_id", type=int)
    recommend.add_argument("--mode", choices=sorted(MODES), required=True)
    recommend.add_argument("--backend", choices=("neo4j", "memory"), default="neo4j")
    recommend.add_argument("--k", type=int, default=10)
    batch=subcommands.add_parser("rank-catalog", help="write Top-K for every catalog seed")
    add_graph_arguments(batch)
    add_connection_arguments(batch)
    batch.add_argument("--mode", choices=sorted(MODES), required=True)
    batch.add_argument("--backend", choices=("neo4j", "memory"), default="neo4j")
    batch.add_argument("--k", type=int, default=10)
    batch.add_argument("--output", type=Path, required=True)
    args=command.parse_args()
    try:
        graph=build_graph(args.catalog, args.triples, args.cast_limit, args.min_confidence)
        if args.command=="load":
            if not args.password:
                raise ValueError("Set --password or NEO4J_PASSWORD.")
            args.backend="neo4j"
            store=open_store(args, graph)
            if args.reset:
                store.reset()
            print(json.dumps({"graph":graph.summary(), "skipped_triples":graph.skipped_triples, "neo4j":store.load(graph)}, indent=2))
            store.close()
            return
        if args.mode=="enhanced" and not args.triples and args.backend=="memory":
            raise ValueError("Enhanced mode in memory needs --triples.")
        store=open_store(args, graph)
        recommender=KnowledgeGraphRecommender(store, MODES[args.mode])
        if args.command=="recommend":
            seed_title=recommender.seed_title(args.movie_id)
            writer=csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(("rank", "movie_id", "title", "score", "explanation"))
            for position, item in enumerate(recommender.recommend(args.movie_id, args.k), 1):
                writer.writerow((position, item.movie_id, item.title, round(item.score, 4), " | ".join(item.explanation_paths(seed_title))))
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as target:
                writer=csv.writer(target, lineterminator="\n")
                writer.writerow(("seed_movie_id", "rank", "movie_id", "score", "explanation"))
                for seed in sorted(graph.titles):
                    seed_title=recommender.seed_title(seed)
                    for position, item in enumerate(recommender.recommend(seed, args.k), 1):
                        writer.writerow((seed, position, item.movie_id, item.score, " | ".join(item.explanation_paths(seed_title))))
        store.close()
    except (OSError, ValueError) as error:
        command.error(str(error))
if __name__=="__main__":
    main()
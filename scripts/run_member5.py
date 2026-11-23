"""Run the Member 5 final comparison of all four systems.

TF-IDF and Item-KNN load from their saved models. The metadata-only and
LLM-enhanced KG rankings come from Member 4's `movie_rec.kg.cli rank-catalog`.
If those ranking files are missing, this script creates them with K=20 using
Member 4's in-memory graph (the same scoring as the Neo4j store, with no
database needed). Use --kg-backend neo4j to rank from a loaded Neo4j graph.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MEMBER4 = PROCESSED / "member4"
TRIPLES = PROCESSED / "member3" / "semantic_triples.csv"
MAX_K = 20
SYSTEMS = (
    ("tfidf", PROCESSED / "tfidf_model.npz"),
    ("itemknn", PROCESSED / "itemknn_model.npz"),
    ("metadata_kg", MEMBER4 / "metadata_kg_rankings.csv"),
    ("llm_kg", MEMBER4 / "llm_kg_rankings.csv"),
)
KG_MODES = {"metadata_kg": "metadata", "llm_kg": "enhanced"}


def run(*arguments):
    print("Running " + " ".join(str(a) for a in arguments), flush=True)
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def existing(path):
    for candidate in (path, path.with_name(path.name + ".gz")):
        if candidate.exists():
            return candidate
    return None


def make_kg_rankings(name, output, backend, triples):
    mode = KG_MODES[name]
    if mode == "enhanced" and not triples.exists():
        print(f"Skipping {name}: Member 3 triples not found at {triples}", flush=True)
        return None
    command = [
        "-m", "movie_rec.kg.cli", "rank-catalog",
        "--catalog", PROCESSED / "catalog.csv",
        "--mode", mode, "--backend", backend,
        "--k", MAX_K, "--output", output,
    ]
    if mode == "enhanced":
        command += ["--triples", triples]
    run(*command)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="optional debug cap on evaluation queries")
    parser.add_argument("--write-query-metrics", action="store_true")
    parser.add_argument("--kg-backend", choices=("memory", "neo4j"), default="memory")
    parser.add_argument("--triples", type=Path, default=TRIPLES)
    parser.add_argument("--rerank", action="store_true",
                        help="rebuild the KG ranking files even if they exist")
    args = parser.parse_args()
    command = [
        "-m", "movie_rec.cli", "final-evaluate",
        "--directory", PROCESSED,
        "--catalog", PROCESSED / "catalog.csv",
        "--output-dir", PROCESSED / "member5",
    ]
    for name, path in SYSTEMS:
        found = existing(path)
        if name in KG_MODES and (found is None or args.rerank):
            found = make_kg_rankings(name, path, args.kg_backend, args.triples)
        if found is None:
            print(f"Skipping {name}: {path} not found", flush=True)
            continue
        command += ["--system", f"{name}={found}"]
    if args.limit is not None:
        command += ["--limit", args.limit]
    if args.write_query_metrics:
        command.append("--write-query-metrics")
    run(*command)


if __name__ == "__main__":
    main()

"""Fit Item-KNN and evaluate it against the Member 1 TF-IDF baseline."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def run(*arguments):
    print("Running " + " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, help="optional debug cap on evaluation queries")
    args = parser.parse_args()
    evaluate = [
        "-m", "movie_rec.cli", "evaluate",
        "--directory", str(PROCESSED),
        "--k", str(args.k),
        "--output-dir", str(PROCESSED),
    ]
    if args.limit is not None:
        evaluate.extend(["--limit", str(args.limit)])
    run(
        "-m", "movie_rec.cli", "train-itemknn",
        "--catalog", str(PROCESSED / "catalog.csv"),
        "--train", str(PROCESSED / "train.csv.gz"),
        "--output", str(PROCESSED / "itemknn_model.npz"),
        "--min-overlap", str(args.min_overlap),
    )
    run(*evaluate)


if __name__ == "__main__":
    main()

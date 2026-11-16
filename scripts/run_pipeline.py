"""Rebuild the full Member 1 handoff from the raw datasets."""

import argparse
import subprocess
import sys
from pathlib import Path

from movie_rec.handoff import seal_handoff

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    print("Running " + " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download or verify the raw inputs first")
    args = parser.parse_args()
    if args.download:
        run("scripts/download_data.py")
    run("-m", "movie_rec.cli", "prepare")
    run("-m", "movie_rec.cli", "export")
    run("-m", "movie_rec.cli", "train-tfidf")
    run("-m", "movie_rec.cli", "rank-catalog", "--all-catalog",
        "--model", "data/processed/tfidf_model.npz",
        "--output", "data/processed/tfidf_catalog_top10.csv")
    seal_handoff(ROOT / "data" / "processed")
    run("-m", "movie_rec.cli", "verify")


if __name__ == "__main__":
    main()

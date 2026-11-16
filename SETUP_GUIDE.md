# Member 1 setup guide

This project implements the Member 1 assignment in your task table. It prepares
TMDB and MovieLens data, creates the chronological 80/20 split and shared
5,000-movie catalog, and provides a fitted TF-IDF recommender.

The downloaded package contains the completed outputs. You can run recommendations
and hand the split files to your teammates immediately after installing Python dependencies.

## Run on Windows

1. Install Python 3.12 if needed. In PowerShell, check it with `py -3.12 --version`.
2. Extract `member1-movie-recommendation.zip`.
3. Open the extracted `movie-recommendation` folder in VS Code. Open its terminal
   using **Terminal → New Terminal**. The terminal should be in the folder
   containing `pyproject.toml`.
4. Run these commands in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e .
.\.venv\Scripts\python.exe -m movie_rec.cli verify
.\.venv\Scripts\python.exe -m movie_rec.cli recommend 1 --model data/processed/tfidf_model.npz --k 10
```

MovieLens ID `1` is Toy Story. The last command prints ten recommended movies
and their cosine similarity scores. Choose other IDs from
`data/processed/catalog.csv`.

These commands call the virtual environment's Python directly, so PowerShell
activation settings do not need adjustment. Installation requires internet access.
The included fitted model runs on a CPU.

For macOS/Linux, create the environment with `python3.12 -m venv .venv` and
replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in the commands.

## What was completed

| Assignment | Output |
| --- | --- |
| Repository and dependencies | Python package, command-line interface, Git ignore rules, pinned dependencies, rebuild scripts, regression tests |
| Clean and link TMDB and MovieLens | 26,894 unique linked movies, cleaned metadata and source-to-canonical movie ID mapping |
| Chronological 80/20 split | 138,292 eligible users, 15,915,724 train ratings, 4,046,913 test ratings |
| Select the shared catalog | Exactly 5,000 movies, sampled by training popularity, with usable overviews |
| TF-IDF recommendations | Saved vectors, vocabulary and IDF weights, reusable Python API, Top-10 results for all 5,000 catalog seeds |

The pipeline started with all 20,000,263 MovieLens 20M ratings. It dropped 33,784
unlinked ratings, collapsed 32 duplicate user/movie ratings after ID canonicalization,
and excluded users with fewer than 20 linked unique ratings.

## Files to hand off

All data files are in `data/processed/`. The `.csv.gz` files are compressed
CSVs that pandas can read directly.

| File | Contents and recipient |
| --- | --- |
| `catalog.csv` | Shared 5,000 movie records with `movie_id`, `tmdb_id`, title, overview, genres, keywords, cast, directors, release date and training popularity. Members 2–5 use these same IDs. |
| `catalog_ids.csv` | Compact shared MovieLens/TMDB ID mapping |
| `train.csv.gz` | Training ratings for Member 2's Item-KNN fitting and evaluation query construction |
| `test.csv.gz` | Held-out ratings for Member 2's final evaluation labels |
| `split_manifest.csv.gz` | Each eligible user's train/test counts and timestamp boundaries |
| `movies_clean.csv.gz` | All 26,894 linked movie records before catalog selection |
| `movie_id_map.csv` | Original MovieLens IDs mapped to canonical MovieLens IDs |
| `tfidf_model.npz` | Fitted TF-IDF model for Members 2 and 5 |
| `tfidf_catalog_top10.csv` | 50,000 recommendation rows, ten per catalog seed |
| `manifest.json` | Data cleaning counts and sampling configuration |
| `checksums.json` | File integrity hashes used by the verify command |

Genres, keywords, cast, and directors in the movie CSVs are JSON arrays.
Use `json.loads()` to read a cell. Every `movie_id` is a canonical MovieLens
ID. The separate `tmdb_id` joins movie metadata to the graph pipeline.

The 80/20 split was made on each eligible user's full linked history before
catalog sampling. The two rating files retain those histories. Member 2 should
filter to `catalog_ids.csv` wherever the experiment requires the common
candidate pool. That filtering can change the train/test proportions and user
history lengths. Keep the existing split assignments.

## Integrate the TF-IDF baseline

Load the model once and reuse it in evaluation or Streamlit.

```python
from movie_rec.tfidf import TfidfRecommender

model = TfidfRecommender.load("data/processed/tfidf_model.npz")
recommendations = model.recommend(seed_movie_id=1, k=10)

for movie_id, title, score in recommendations:
    print(movie_id, title, score)
```

Recommendations are tuples of `movie_id, title, cosine_score`, sorted by
descending score. The seed is excluded. Other training movies remain eligible
by default. If the team agrees on removing a user's previously seen movies
for every method, pass them through `exclude=seen_movie_ids`.

Member 2 can read and filter the training file in chunks.

```python
import pandas as pd

catalog = pd.read_csv("data/processed/catalog.csv")
movie_ids = set(catalog["movie_id"])
parts = []
for chunk in pd.read_csv("data/processed/train.csv.gz", chunksize=1_000_000):
    parts.append(chunk[chunk["movie_id"].isin(movie_ids)])
train = pd.concat(parts, ignore_index=True)
```

The presentation defines positive seeds as training movies rated at least 4.0,
and relevant test movies as held-out movies rated at least 4.0 by the same user.
Member 2 owns the common query policy and ranking metrics. The included Top-10
file covers every catalog seed so it can support that shared evaluation.

## Rebuild everything from the original data

From the project folder, run this in PowerShell after installation.

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --download
```

The command downloads or verifies the five source CSVs, cleans and links them,
recreates the split and catalog, exports the handoff files, fits the model,
writes Top-10 rankings, and verifies the output hashes. Allow several minutes
and around 4 GB of free disk space. Runtime depends on disk and CPU speed.

The downloader uses official MovieLens 20M and a fixed revision of a public
mirror of The Movies Dataset. Every downloaded file is checked against its
recorded checksum. Sources and hashes are documented in `SOURCES.md`.
If a download endpoint is unavailable, obtain the five files from the documented
sources and use the same command without `--download`.

The rebuilt `data/processed/experiment.sqlite` contains the full cleaned data
and SQL views. The download package carries the exported split CSVs. The
SQLite database is recreated by the rebuild command.

To regenerate only the fitted model and rankings, run these commands.

```powershell
.\.venv\Scripts\python.exe -m movie_rec.cli train-tfidf
.\.venv\Scripts\python.exe -m movie_rec.cli rank-catalog --all-catalog --model data/processed/tfidf_model.npz --output data/processed/tfidf_catalog_top10.csv
```

The packaged checksums describe the delivered output files. A full rebuild
updates the checksums automatically after all stages complete.

## Experimental choices

- The user threshold is at least 20 linked, unique ratings.
- Duplicate user/movie ratings keep the latest valid event. Equal timestamps
  keep the later source row.
- The first floor(0.8 × n) ratings per user form training. Sort order is
  timestamp, then canonical movie ID. Equal timestamps can occur across the
  boundary. This is a per-user chronological split.
- A movie with multiple MovieLens IDs uses the lowest linked MovieLens ID.
  The mapping file preserves the aliases. Titles and years are never used to
  guess a match.
- Catalog sampling uses random seed 42 and training-rating count bands
  0, 1–9, 10–99, 100–999, 1,000–9,999, and 10,000+. It assigns approximately
  equal places to the bands, redistributing places when a band is exhausted.
  The resulting counts are 928 in each of the first five bands and 360 in the last.
- The catalog requires a nonempty title and overview. The 928 movies with zero
  training ratings are intentionally retained for sparse-item analysis.
- TF-IDF uses overview, genre names, and keyword names with unigrams and
  bigrams, English stop words, and L2 normalization. Cosine ties use ascending
  movie ID. Cast and directors are available for the graph methods.
- TF-IDF fits static metadata for the shared catalog. Rating labels never enter
  TF-IDF fitting or similarity scoring. TMDB's 2017 metadata snapshot is later
  than the MovieLens 20M rating period. Record that retrospective-data
  limitation in the report.

## Repository setup and checks

The source was initialized and committed in a local Git repository during
preparation. To start a Git repository from the extracted source package, run
the following from its root.

```powershell
git init -b main
git add .
git commit -m "Complete Member 1 data preparation and TF-IDF baseline"
```

A remote repository URL has not been supplied. Add your team's remote when it
is available. The included `.gitignore` excludes datasets, generated output,
and the virtual environment. Keep source code in Git and retain the data handoff
in the team's approved project storage.

Run regression checks with the following command.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The package includes `VALIDATION.md` with the observed full-data checks.

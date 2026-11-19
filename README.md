# Semantic Movie Recommender

CSE 573 · Group 13

This repository implements data preparation, the TF-IDF baseline, Item-KNN,
semantic triple extraction, and a shared evaluation protocol for the movie
recommendation project. It links
MovieLens 20M ratings to TMDB metadata, creates chronological per-user 80/20
splits, and selects a shared catalog of 5,000 movies using training popularity.

## Setup

Use Python 3.12. From the project directory, run the following on Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e .
```

On macOS/Linux, use `python3.12` to create the environment and
`.venv/bin/python` for subsequent commands.

## Prepare the data and model

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --download
```

This downloads and verifies the source files, cleans and links the datasets,
creates the split and catalog, exports the data, fits TF-IDF, and generates
Top-10 rankings. Outputs are written to `data/processed/`.

If prepared data and model files are already available, place them in
`data/processed/` and run the following commands.

```powershell
.\.venv\Scripts\python.exe -m movie_rec.cli verify
.\.venv\Scripts\python.exe -m movie_rec.cli recommend 1 --model data/processed/tfidf_model.npz --k 10
```

MovieLens ID 1 is Toy Story. Recommendations include movie IDs, titles, and
cosine similarity scores.

## Item-KNN and shared evaluation

Member 2 reuses the Member 1 catalog, split, seed queries, and labels. It does
not rebuild IDs, the split, or the catalog. With prepared files in
`data/processed/`:

```powershell
.\.venv\Scripts\python.exe scripts/run_member2.py
```

Or run the steps separately:

```powershell
.\.venv\Scripts\python.exe -m movie_rec.cli train-itemknn --min-overlap 5
.\.venv\Scripts\python.exe -m movie_rec.cli recommend 1 --model data/processed/itemknn_model.npz --k 10
.\.venv\Scripts\python.exe -m movie_rec.cli evaluate --k 10
```

`evaluate` scores TF-IDF and Item-KNN on the same queries and writes
`baseline_metrics.csv`, `baseline_metrics.json`, and
`baseline_query_metrics.csv`. Pass `--limit` only when debugging.

Item-KNN builds a sparse user–movie matrix from **training** ratings on catalog
movies, computes item cosine similarity, and zeros pairs with fewer than
`min_overlap` co-raters (default 5). Test ratings are not used for fitting.
The seed movie is always excluded. Previously rated training movies are not
also filtered: Member 1 and the project slides only document excluding the
seed, while requiring the same catalog, seeds, labels, and Top-K.

Evaluation queries are Member 1 `seed_queries`: catalog movies a user rated
≥ 4.0 in training, limited to users who have at least one catalog test rating
≥ 4.0. Ground truth is `evaluation_labels`. Metrics are Precision@K, Recall@K,
binary NDCG@K, and Hit Rate@K, macro-averaged over queries. K is configurable;
K=10 is the primary reported cutoff.

Full MovieLens 20M evaluation (5,404,341 queries, seed excluded only):

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.00808 | 0.00568 | 0.00963 | 0.07429 |
| Item-KNN | 0.03480 | 0.03024 | 0.03958 | 0.25181 |

These numbers are reproduced by `scripts/run_member2.py`. Metric CSVs, models, and
`data/processed/` stay gitignored.

## Semantic triples (Member 3)

Member 3 reads Member 1's `data/processed/catalog.csv` and does not create new
movie IDs. Copy that file from Member 1 before running the pilot.
`catalog_ids.csv` has no overviews. The model receives only the overview text.

The prompt is `prompts/extract_triples.txt`. Keep the `{overview}` placeholder.
A pilot run selects 50 distinct movies with nonempty overviews (seed 42) and
saves them to `data/processed/member3/pilot_movie_ids.csv`. Later pilot runs
reuse that file. The prompt version is the first 12 hex characters of the
prompt text's SHA-256. Results for that version go to:

`data/processed/member3/pilot/qwen2.5-3b/<prompt_version>/`

`pilot_review.csv` has the title, overview, predicate, model label, normalized
label, and quote. `rejections.csv` records why a triple was dropped.
`extraction_status.csv` records empty overviews and model failures. Editing the
prompt and rerunning the pilot leaves the previous version's directory in place.

Labels are Unicode-normalized, lowercased, and stripped of punctuation. A
leading a/an/the is removed. Predicate case is canonicalized to `HAS_THEME`,
`HAS_SETTING`, or `HAS_CONCEPT`. A quote is kept only when it is an exact
contiguous span of the overview. The same movie, predicate, and normalized
label are not stored twice.

Ollama must be running locally with `qwen2.5:3b` (`ollama pull qwen2.5:3b`).
From the repository root:

```powershell
.\.venv\Scripts\python.exe -m movie_rec.cli extract-pilot --catalog data/processed/catalog.csv --prompt prompts/extract_triples.txt --model qwen2.5:3b --n 50 --seed 42 --output-dir data/processed/member3
```

After the pilot looks right, the resumable full-catalog command writes Member 4's
`semantic_triples.csv` with `movie_id`, `tmdb_id`, `predicate`,
`normalized_label`, `evidence`, `model`, and `prompt_version`. Movies with a
blank overview or a failed extraction stay in `extraction_status.csv` and are
omitted from the triples file. Run the same command again to continue from
`checkpoints.jsonl`. Add `--retry-failed` to repeat failed movies.

```powershell
.\.venv\Scripts\python.exe -m movie_rec.cli extract-catalog --catalog data/processed/catalog.csv --prompt prompts/extract_triples.txt --model qwen2.5:3b --output-dir data/processed/member3
```

## Method

Users must have at least 20 linked, unique ratings. Their earliest
floor(0.8 × n) ratings form training, with the remaining ratings held out for
testing. Equal timestamps are ordered by canonical movie ID.

Catalog sampling uses seed 42 and training-rating count bands of 0, 1–9,
10–99, 100–999, 1,000–9,999, and 10,000+. Movies require a title and overview.
The split precedes catalog filtering, so filtering to the catalog can change
each user's train/test proportions.

TF-IDF uses overviews, genres, and keywords, with unigrams and bigrams,
English stop-word removal, and L2 normalization. Item-KNN uses cosine
similarity of training rating vectors with a minimum co-rating threshold.
Recommendations exclude the seed movie and resolve equal scores by movie ID.

## Outputs

| File | Contents |
| --- | --- |
| catalog.csv | Shared movie metadata |
| catalog_ids.csv | Canonical MovieLens and TMDB IDs |
| train.csv.gz and test.csv.gz | Full linked rating partitions |
| split_manifest.csv.gz | Per-user counts and timestamp boundaries |
| movie_id_map.csv | Source-to-canonical MovieLens IDs |
| movies_clean.csv.gz | Cleaned linked metadata |
| tfidf_model.npz | Fitted TF-IDF model |
| tfidf_catalog_top10.csv | Ten recommendations per catalog seed |
| itemknn_model.npz | Fitted Item-KNN model (Member 2; not part of Member 1 checksums) |
| baseline_metrics.csv / baseline_metrics.json | TF-IDF vs Item-KNN metrics |
| baseline_query_metrics.csv | Per-query scores for later comparisons |
| manifest.json and checksums.json | Preparation counts and output integrity hashes |

Raw data, generated outputs, and virtual environments are excluded from Git.
See [SOURCES.md](SOURCES.md) for data provenance and usage terms, and
[VALIDATION.md](VALIDATION.md) for the completed checks.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

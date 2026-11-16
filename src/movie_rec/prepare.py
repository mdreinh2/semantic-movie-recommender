"""Build the linked MovieLens data and the shared experiment catalog."""

import ast
import csv
import json
import random
import sqlite3
import tempfile
from collections import Counter, defaultdict, deque
from contextlib import closing
from datetime import date
from pathlib import Path

MOVIE_COLUMNS = (
    "movie_id", "tmdb_id", "title", "overview", "release_date",
    "genres", "keywords", "cast", "directors",
)
BAND_BOUNDS = (0, 9, 99, 999, 9999)
BAND_NAMES = ("0", "1-9", "10-99", "100-999", "1000-9999", "10000+")


def rows(path, required):
    # Some TMDB credit fields are larger than the default CSV field limit.
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        yield from reader


def positive_id(value):
    value = (value or "").strip()
    return int(value) if value.isascii() and value.isdecimal() and int(value) > 0 else None


def clean_text(value):
    return " ".join((value or "").split())


def parse_records(value, stats):
    if not (value or "").strip():
        return []
    try:
        try:
            records = json.loads(value)
        except ValueError:
            records = ast.literal_eval(value)
        if not isinstance(records, list):
            raise ValueError("Expected a list")
    except (ValueError, SyntaxError, TypeError):
        stats["invalid_metadata_lists"] += 1
        return []
    return [record for record in records if isinstance(record, dict)]


def names(records, job=None):
    return list(dict.fromkeys(
        clean_text(record["name"]) for record in records
        if isinstance(record.get("name"), str) and record["name"].strip()
        and (job is None or record.get("job") == job)
    ))


def release_date(value):
    try:
        return date.fromisoformat(value or "").isoformat()
    except ValueError:
        return ""


def read_metadata(tmdb_dir, wanted, stats):
    movies = {}
    fields = ("id", "title", "overview", "release_date", "genres")
    for row in rows(tmdb_dir / "movies_metadata.csv", fields):
        tmdb_id = positive_id(row["id"])
        if tmdb_id is None:
            stats["invalid_metadata_ids"] += 1
            continue
        if tmdb_id not in wanted:
            continue
        movie = (
            clean_text(row["title"]), clean_text(row["overview"]),
            release_date(row["release_date"]), names(parse_records(row["genres"], stats)),
        )
        if tmdb_id in movies:
            stats["duplicate_metadata_rows"] += 1
        # Prefer complete title/overview fields, keeping source order for ties.
        if tmdb_id not in movies or (bool(movie[0]), bool(movie[1])) > (
            bool(movies[tmdb_id][0]), bool(movies[tmdb_id][1])
        ):
            movies[tmdb_id] = movie

    keywords = defaultdict(list)
    for row in rows(tmdb_dir / "keywords.csv", ("id", "keywords")):
        tmdb_id = positive_id(row["id"])
        if tmdb_id in movies:
            keywords[tmdb_id].extend(names(parse_records(row["keywords"], stats)))

    casts, directors = defaultdict(list), defaultdict(list)
    for row in rows(tmdb_dir / "credits.csv", ("id", "cast", "crew")):
        tmdb_id = positive_id(row["id"])
        if tmdb_id in movies:
            casts[tmdb_id].extend(names(parse_records(row["cast"], stats)))
            directors[tmdb_id].extend(names(parse_records(row["crew"], stats), "Director"))

    return {
        tmdb_id: (title, overview, released, *(
            json.dumps(list(dict.fromkeys(values)), ensure_ascii=False)
            for values in (genres, keywords[tmdb_id], casts[tmdb_id], directors[tmdb_id])
        ))
        for tmdb_id, (title, overview, released, genres) in movies.items()
        if title
    }


def link_movies(links_path, tmdb_dir, stats):
    links = defaultdict(set)
    for row in rows(links_path, ("movieId", "tmdbId")):
        movie_id, tmdb_id = positive_id(row["movieId"]), positive_id(row["tmdbId"])
        if movie_id and tmdb_id:
            links[movie_id].add(tmdb_id)
        else:
            stats["invalid_or_missing_link_rows"] += 1

    stats["ambiguous_movielens_ids"] = sum(len(ids) > 1 for ids in links.values())
    links = {movie_id: next(iter(ids)) for movie_id, ids in links.items() if len(ids) == 1}
    metadata = read_metadata(tmdb_dir, set(links.values()), stats)
    canonical = {}
    movie_rows = []
    aliases = {}
    for movie_id, tmdb_id in sorted(links.items()):
        if tmdb_id not in metadata:
            stats["links_without_usable_metadata"] += 1
            continue
        if tmdb_id not in canonical:
            canonical[tmdb_id] = movie_id
            movie_rows.append((movie_id, tmdb_id, *metadata[tmdb_id]))
        aliases[movie_id] = canonical[tmdb_id]
    stats["alias_movielens_ids"] = len(aliases) - len(movie_rows)
    return movie_rows, aliases


def load_ratings(db, ratings_path, aliases, stats):
    batch = []
    insert = """
        INSERT INTO linked_ratings VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET
            rating = excluded.rating, timestamp = excluded.timestamp
        WHERE excluded.timestamp >= linked_ratings.timestamp
    """
    for row in rows(ratings_path, ("userId", "movieId", "rating", "timestamp")):
        stats["input_ratings"] += 1
        try:
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            rating = float(row["rating"])
            timestamp = int(row["timestamp"])
            if user_id <= 0 or movie_id <= 0 or timestamp < 0 or not (
                0.5 <= rating <= 5 and (rating * 2).is_integer()
            ):
                raise ValueError
        except (ValueError, TypeError):
            stats["invalid_ratings"] += 1
            continue
        movie = aliases.get(movie_id)
        if movie is None:
            stats["unlinked_ratings"] += 1
            continue
        batch.append((user_id, movie, rating, timestamp))
        if len(batch) == 50_000:
            db.executemany(insert, batch)
            batch.clear()
    db.executemany(insert, batch)
    linked = db.execute("SELECT COUNT(*) FROM linked_ratings").fetchone()[0]
    stats["linked_ratings"] = linked
    stats["duplicate_ratings"] = (
        stats["input_ratings"] - stats["invalid_ratings"] - stats["unlinked_ratings"] - linked
    )


def select_catalog(popularity, size, seed):
    rng = random.Random(seed)
    bands = [[] for _ in BAND_NAMES]
    for movie_id, count in popularity:
        band = sum(count > bound for bound in BAND_BOUNDS)
        bands[band].append((movie_id, count, BAND_NAMES[band]))
    for band in bands:
        rng.shuffle(band)
    active = [deque(band) for band in bands if band]
    selected = []
    while active and len(selected) < size:
        for band in active:
            selected.append(band.popleft())
            if len(selected) == size:
                break
        active = [band for band in active if band]
    return sorted(selected)


SCHEMA = """
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY, tmdb_id INTEGER UNIQUE, title TEXT, overview TEXT,
    release_date TEXT, genres TEXT, keywords TEXT, cast TEXT, directors TEXT
);
CREATE TABLE movie_id_map (
    source_movie_id INTEGER PRIMARY KEY, movie_id INTEGER NOT NULL REFERENCES movies
);
CREATE TABLE linked_ratings (
    user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER,
    PRIMARY KEY (user_id, movie_id)
) WITHOUT ROWID;
CREATE TABLE ratings (
    user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER, split TEXT,
    PRIMARY KEY (user_id, movie_id)
) WITHOUT ROWID;
CREATE TABLE catalog (
    movie_id INTEGER PRIMARY KEY REFERENCES movies, train_count INTEGER, popularity_band TEXT
);
CREATE VIEW train_ratings AS
    SELECT user_id, movie_id, rating, timestamp FROM ratings WHERE split = 'train';
CREATE VIEW test_ratings AS
    SELECT user_id, movie_id, rating, timestamp FROM ratings WHERE split = 'test';
CREATE VIEW evaluation_labels AS
    SELECT user_id, movie_id FROM test_ratings JOIN catalog USING (movie_id) WHERE rating >= 4.0;
CREATE VIEW seed_queries AS
    SELECT r.user_id, r.movie_id AS seed_movie_id
    FROM train_ratings r JOIN catalog USING (movie_id)
    WHERE r.rating >= 4.0 AND EXISTS (
        SELECT 1 FROM evaluation_labels l WHERE l.user_id = r.user_id
    );
"""

SPLIT = """
WITH eligible AS (
    SELECT user_id, COUNT(*) AS n
    FROM linked_ratings GROUP BY user_id HAVING COUNT(*) >= 20
), ordered AS (
    SELECT r.*, e.n, ROW_NUMBER() OVER (
        PARTITION BY user_id ORDER BY timestamp, movie_id
    ) AS position
    FROM linked_ratings r JOIN eligible e USING (user_id)
)
INSERT INTO ratings
SELECT user_id, movie_id, rating, timestamp,
       CASE WHEN position <= 4*n/5 THEN 'train' ELSE 'test' END
FROM ordered;
"""


def prepare(movielens_dir: Path, tmdb_dir: Path, output: Path, size=5000, seed=42):
    if size < 1:
        raise ValueError("Catalog size must be positive.")
    stats = Counter({key: 0 for key in (
        "invalid_metadata_lists", "invalid_metadata_ids", "duplicate_metadata_rows",
        "invalid_or_missing_link_rows", "links_without_usable_metadata",
        "input_ratings", "invalid_ratings", "unlinked_ratings",
    )})
    movies, aliases = link_movies(movielens_dir / "links.csv", tmdb_dir, stats)
    if not movies:
        raise ValueError("No usable MovieLens–TMDB matches.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as tmp:
        staging = Path(tmp)
        with closing(sqlite3.connect(staging / "experiment.sqlite")) as db, db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.executescript(SCHEMA)
            db.executemany("INSERT INTO movies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", movies)
            db.executemany("INSERT INTO movie_id_map VALUES (?, ?)", aliases.items())
            load_ratings(db, movielens_dir / "ratings.csv", aliases, stats)
            db.execute(SPLIT)
            stats["eligible_users"] = db.execute("SELECT COUNT(DISTINCT user_id) FROM ratings").fetchone()[0]
            if not stats["eligible_users"]:
                raise ValueError("No users have at least 20 linked, deduplicated ratings.")
            for split, count in db.execute("SELECT split, COUNT(*) FROM ratings GROUP BY split"):
                stats[f"{split}_ratings"] = count
            db.execute("DROP TABLE linked_ratings")
            popularity = db.execute("""
                WITH counts AS (
                    SELECT movie_id, COUNT(*) AS n FROM train_ratings GROUP BY movie_id
                )
                SELECT m.movie_id, COALESCE(c.n, 0)
                FROM movies m LEFT JOIN counts c USING (movie_id)
                WHERE m.overview <> '' ORDER BY m.movie_id
            """)
            stats["movies_with_overviews"] = db.execute(
                "SELECT COUNT(*) FROM movies WHERE overview <> ''"
            ).fetchone()[0]
            catalog = select_catalog(popularity, size, seed)
            db.executemany("INSERT INTO catalog VALUES (?, ?, ?)", catalog)
            with (staging / "catalog.csv").open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow((*MOVIE_COLUMNS, "train_count", "popularity_band"))
                writer.writerows(db.execute(
                    "SELECT m.*, c.train_count, c.popularity_band "
                    "FROM movies m JOIN catalog c USING (movie_id) ORDER BY movie_id"
                ))
            stats["linked_movies"] = len(movies)
            stats["catalog_movies"] = len(catalog)
        with closing(sqlite3.connect(staging / "experiment.sqlite")) as db:
            db.execute("VACUUM")
        manifest = {
            "counts": dict(sorted(stats.items())),
            "catalog_seed": seed,
            "requested_catalog_size": size,
            "catalog_by_popularity_band": dict(Counter(row[2] for row in catalog)),
            "min_ratings_per_user": 20,
            "positive_rating_threshold": 4.0,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in ("experiment.sqlite", "catalog.csv", "manifest.json"):
            (staging / name).replace(output / name)
    return manifest

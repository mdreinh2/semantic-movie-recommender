"""Extract and validate theme, setting, and concept triples from overviews.

The editable prompt is prompts/extract_triples.txt. `{overview}` is the only
placeholder, and it is the only catalog text sent to the model. Pilot movie IDs
are written once to ``pilot_movie_ids.csv``. Each prompt hash is stored in its
own directory, so refining the prompt keeps the earlier pilot results.
"""

import csv
import hashlib
import json
import random
import re
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .handoff import sha256 as file_sha256
from .prepare import rows

PREDICATES = ("HAS_THEME", "HAS_SETTING", "HAS_CONCEPT")
_ARTICLES = {"a", "an", "the"}
_MISSING_CATALOG = (
    "Catalog not found at {path}. Copy catalog.csv from Member 1 into "
    "data/processed/catalog.csv. That file includes movie_id, tmdb_id, title, "
    "and overview. catalog_ids.csv does not contain overviews."
)

REVIEW_COLUMNS = (
    "movie_id", "tmdb_id", "title", "overview", "predicate",
    "label", "normalized_label", "quote",
)
REJECTION_COLUMNS = (
    "movie_id", "tmdb_id", "title", "predicate", "label",
    "normalized_label", "quote", "reason",
)
STATUS_COLUMNS = (
    "movie_id", "tmdb_id", "title", "status", "n_accepted", "n_rejected",
    "error", "model", "prompt_version",
)
TRIPLE_COLUMNS = (
    "movie_id", "tmdb_id", "predicate", "normalized_label", "evidence",
    "model", "prompt_version",
)


class ModelError(Exception):
    """The local model request failed or returned an unusable response."""


def load_catalog(path):
    """Load Member 1 catalog rows, keeping the first copy of each movie_id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_MISSING_CATALOG.format(path=path))
    movies = []
    seen = set()
    for row in rows(path, ("movie_id", "tmdb_id", "title", "overview")):
        movie_id = required_id(row["movie_id"], "movie_id")
        if movie_id in seen:
            continue
        seen.add(movie_id)
        movies.append({
            "movie_id": movie_id,
            "tmdb_id": required_id(row["tmdb_id"], "tmdb_id"),
            "title": row.get("title") or "",
            "overview": row.get("overview") or "",
        })
    if not movies:
        raise ValueError(f"{path} does not contain any movies.")
    return movies


def required_id(value, field):
    text = (value or "").strip()
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return int(text)


def has_overview(value):
    return bool((value or "").strip())


def prompt_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_version(text):
    """Short stable ID for a prompt. Editing the file produces a new ID."""
    return prompt_sha256(text)[:12]


def model_dirname(model):
    """Return a path segment that is safe on Windows."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    return token or "model"


def output_directory(output_dir, kind, model, template):
    return Path(output_dir) / kind / model_dirname(model) / prompt_version(template)


def read_prompt(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found at {path}.")
    text = path.read_text(encoding="utf-8-sig")
    if "{overview}" not in text:
        raise ValueError(f"{path} must contain the {{overview}} placeholder.")
    return text


def render_prompt(template, overview):
    """Insert the overview and leave every other character unchanged."""
    if "{overview}" not in template:
        raise ValueError("Prompt must contain the {overview} placeholder.")
    return template.replace("{overview}", overview)


def normalize_label(label):
    """Fold case and punctuation so Member 4 can match equivalent labels."""
    text = unicodedata.normalize("NFKC", label or "")
    text = text.casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    while tokens and tokens[0] in _ARTICLES:
        del tokens[0]
    return " ".join(tokens)


def parse_triples_payload(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_json") from error
    if not isinstance(data, dict):
        raise ValueError("invalid_json")
    if "triples" not in data:
        raise ValueError("missing_triples")
    if not isinstance(data["triples"], list):
        raise ValueError("triples_not_list")
    return data["triples"]


def quote_is_exact(overview, quote):
    """True when quote is a contiguous overview span, not just letters inside a word."""
    if not quote:
        return False
    start = 0
    while True:
        index = overview.find(quote, start)
        if index < 0:
            return False
        before = overview[index - 1] if index else ""
        end = index + len(quote)
        after = overview[end] if end < len(overview) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = index + 1


def _rejection(predicate, label, quote, reason, normalized=""):
    return {
        "predicate": predicate.strip() if isinstance(predicate, str) else "",
        "label": label.strip() if isinstance(label, str) else "",
        "normalized_label": normalized,
        "evidence": quote.strip() if isinstance(quote, str) else "",
        "reason": reason,
    }


def validate_triples(overview, triples):
    """Keep triples with an allowed predicate and an exact supporting quote."""
    accepted = []
    rejected = []
    seen = set()
    for item in triples:
        if not isinstance(item, dict):
            rejected.append(_rejection("", "", "", "invalid_item"))
            continue
        raw_predicate = item.get("predicate", "")
        raw_label = item.get("label", "")
        raw_quote = item.get("quote", "")
        predicate = raw_predicate.strip().upper() if isinstance(raw_predicate, str) else ""
        if predicate not in PREDICATES:
            rejected.append(_rejection(raw_predicate, raw_label, raw_quote, "invalid_predicate"))
            continue
        if not isinstance(raw_label, str) or not raw_label.strip():
            rejected.append(_rejection(predicate, raw_label, raw_quote, "missing_label"))
            continue
        normalized = normalize_label(raw_label)
        if not normalized:
            rejected.append(_rejection(
                predicate, raw_label, raw_quote, "label_empty_after_normalization",
            ))
            continue
        if not isinstance(raw_quote, str) or not raw_quote.strip():
            rejected.append(_rejection(predicate, raw_label, raw_quote, "missing_quote", normalized))
            continue
        quote = raw_quote.strip()
        if quote not in overview:
            rejected.append(_rejection(
                predicate, raw_label, quote, "quote_not_in_overview", normalized,
            ))
            continue
        if not quote_is_exact(overview, quote):
            rejected.append(_rejection(
                predicate, raw_label, quote, "quote_not_exact_span", normalized,
            ))
            continue
        key = (predicate, normalized)
        if key in seen:
            rejected.append(_rejection(predicate, raw_label, quote, "duplicate", normalized))
            continue
        seen.add(key)
        accepted.append({
            "predicate": predicate,
            "normalized_label": normalized,
            "raw_label": raw_label.strip(),
            "evidence": quote,
        })
    return accepted, rejected


def extract_movie(movie, template, complete):
    """Extract one movie. Blank overviews are recorded and are not sent to the model."""
    base = {
        "movie_id": movie["movie_id"],
        "tmdb_id": movie["tmdb_id"],
        "title": movie["title"],
        "status": "ok",
        "error": "",
        "accepted": [],
        "rejected": [],
    }
    overview = movie["overview"] or ""
    if not has_overview(overview):
        base["status"] = "missing_overview"
        return base
    try:
        payload = parse_triples_payload(complete(render_prompt(template, overview)))
    except ModelError as error:
        base["status"] = "extraction_failed"
        base["error"] = f"model_error: {error}"
        return base
    except ValueError as error:
        base["status"] = "extraction_failed"
        base["error"] = str(error)
        return base
    accepted, rejected = validate_triples(overview, payload)
    base["accepted"] = accepted
    base["rejected"] = rejected
    return base


class OllamaClient:
    """Call a local Ollama chat endpoint. The prompt is the only message."""

    def __init__(self, model, base_url="http://127.0.0.1:11434", timeout=180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ModelError(f"Ollama HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise ModelError(f"Ollama request failed: {reason}") from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ModelError("Ollama response was not valid JSON.") from error
        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError("Ollama response did not include message content.")
        return content


def select_pilot_movie_ids(movies, n, seed):
    """Choose distinct movies that have overviews. The draw depends only on seed."""
    if n < 1:
        raise ValueError("Pilot size must be positive.")
    eligible = sorted({
        movie["movie_id"] for movie in movies if has_overview(movie["overview"])
    })
    rng = random_for(seed)
    if len(eligible) <= n:
        chosen = eligible[:]
        rng.shuffle(chosen)
        return chosen
    return rng.sample(eligible, n)


def random_for(seed):
    return random.Random(seed)


def resolve_pilot_ids(path, movies, n, seed):
    """Reuse a saved ID file so prompt revisions score the same movies."""
    path = Path(path)
    if path.is_file():
        movie_ids, stored_seed, stored_n = load_pilot_ids(path)
        if not movie_ids:
            raise ValueError(f"{path} does not contain pilot movie IDs.")
        return movie_ids, True, stored_seed, stored_n
    movie_ids = select_pilot_movie_ids(movies, n, seed)
    save_pilot_ids(path, movie_ids, seed, n)
    return movie_ids, False, seed, n


def load_pilot_ids(path):
    movie_ids = []
    seen = set()
    stored_seed = None
    stored_n = None
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "movie_id" not in reader.fieldnames:
            raise ValueError(f"{path} is missing a movie_id column.")
        for row in reader:
            movie_id = required_id(row["movie_id"], "movie_id")
            if movie_id in seen:
                continue
            seen.add(movie_id)
            movie_ids.append(movie_id)
            if stored_seed is None and (row.get("seed") or "").strip():
                stored_seed = int(row["seed"])
            if stored_n is None and (row.get("requested_n") or "").strip():
                stored_n = int(row["requested_n"])
    return movie_ids, stored_seed, stored_n


def save_pilot_ids(path, movie_ids, seed, requested_n):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("movie_id", "seed", "requested_n"))
        writer.writerows((movie_id, seed, requested_n) for movie_id in movie_ids)


def movies_for_ids(movies, movie_ids):
    index = {movie["movie_id"]: movie for movie in movies}
    missing = [movie_id for movie_id in movie_ids if movie_id not in index]
    if missing:
        preview = ", ".join(str(movie_id) for movie_id in missing[:10])
        raise ValueError(f"Pilot IDs are not in the catalog: {preview}")
    return [index[movie_id] for movie_id in movie_ids]


def load_checkpoints(path):
    """Return the latest record per movie. A truncated final line is discarded."""
    path = Path(path)
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Corrupt checkpoint line in {path}.") from None
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in parsed),
                encoding="utf-8",
            )
            break
    latest = {}
    for row in parsed:
        latest[int(row["movie_id"])] = row
    return latest


def write_csv(path, header, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(records)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _status_row(movie, record, model, version):
    if record is None:
        return (movie["movie_id"], movie["tmdb_id"], movie["title"], "not_started", 0, 0, "", model, version)
    accepted = record.get("accepted") or []
    rejected = record.get("rejected") or []
    return (
        movie["movie_id"], movie["tmdb_id"], movie["title"], record.get("status") or "",
        len(accepted), len(rejected), record.get("error") or "",
        record.get("model") or model, record.get("prompt_version") or version,
    )


def _rejection_rows(movie, record):
    rows = []
    for item in record.get("rejected") or []:
        rows.append((
            movie["movie_id"], movie["tmdb_id"], movie["title"],
            item.get("predicate") or "", item.get("label") or "",
            item.get("normalized_label") or "", item.get("evidence") or "",
            item.get("reason") or "",
        ))
    if record.get("status") == "extraction_failed":
        rows.append((
            movie["movie_id"], movie["tmdb_id"], movie["title"],
            "", "", "", "", record.get("error") or "extraction_failed",
        ))
    return rows


def export_pilot(destination, selected, results, model, version):
    by_id = {result["movie_id"]: result for result in results}
    review, rejections, statuses = [], [], []
    for movie in selected:
        result = by_id[movie["movie_id"]]
        record = {**result, "model": model, "prompt_version": version}
        statuses.append(_status_row(movie, record, model, version))
        rejections.extend(_rejection_rows(movie, record))
        for triple in result["accepted"]:
            review.append((
                movie["movie_id"], movie["tmdb_id"], movie["title"], movie["overview"],
                triple["predicate"], triple["raw_label"], triple["normalized_label"],
                triple["evidence"],
            ))
    write_csv(destination / "pilot_review.csv", REVIEW_COLUMNS, review)
    write_csv(destination / "rejections.csv", REJECTION_COLUMNS, rejections)
    write_csv(destination / "extraction_status.csv", STATUS_COLUMNS, statuses)
    return {
        "review_csv": str(destination / "pilot_review.csv"),
        "rejections_csv": str(destination / "rejections.csv"),
        "status_csv": str(destination / "extraction_status.csv"),
        "n_accepted": len(review),
        "n_rejected": len(rejections),
        "n_missing_overview": sum(row[3] == "missing_overview" for row in statuses),
        "n_failed": sum(row[3] == "extraction_failed" for row in statuses),
        "n_ok": sum(row[3] == "ok" for row in statuses),
    }


def export_catalog(destination, movies, latest, model, version):
    ordered = sorted(movies, key=lambda movie: movie["movie_id"])
    triples, rejections, statuses = [], [], []
    for movie in ordered:
        record = latest.get(movie["movie_id"])
        statuses.append(_status_row(movie, record, model, version))
        if not record:
            continue
        rejections.extend(_rejection_rows(movie, record))
        if record.get("status") != "ok":
            continue
        for triple in record.get("accepted") or []:
            triples.append((
                movie["movie_id"], movie["tmdb_id"], triple["predicate"],
                triple["normalized_label"], triple["evidence"],
                record.get("model") or model, record.get("prompt_version") or version,
            ))
    triples.sort(key=lambda row: (row[0], row[2], row[3], row[4]))
    write_csv(destination / "semantic_triples.csv", TRIPLE_COLUMNS, triples)
    write_csv(destination / "rejections.csv", REJECTION_COLUMNS, rejections)
    write_csv(destination / "extraction_status.csv", STATUS_COLUMNS, statuses)
    return {
        "triples_csv": str(destination / "semantic_triples.csv"),
        "rejections_csv": str(destination / "rejections.csv"),
        "status_csv": str(destination / "extraction_status.csv"),
        "n_accepted": len(triples),
        "n_rejected": len(rejections),
        "n_missing_overview": sum(row[3] == "missing_overview" for row in statuses),
        "n_failed": sum(row[3] == "extraction_failed" for row in statuses),
        "n_ok": sum(row[3] == "ok" for row in statuses),
        "n_not_started": sum(row[3] == "not_started" for row in statuses),
    }


def _provenance(command, catalog, prompt, template, model, base_url, extra):
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "member": 3,
        "command": command,
        "model": model,
        "ollama_url": base_url,
        "prompt_path": str(Path(prompt)),
        "prompt_sha256": prompt_sha256(template),
        "prompt_version": prompt_version(template),
        "catalog_path": str(Path(catalog)),
        "catalog_sha256": file_sha256(catalog),
        "created_at": now,
    }
    payload.update(extra)
    return payload


def _log(verbose, message):
    if verbose:
        print(message, flush=True)


def run_pilot(
    catalog, prompt, output_dir, model="qwen2.5:3b", n=50, seed=42,
    base_url="http://127.0.0.1:11434", timeout=180, ids_path=None,
    complete=None, verbose=True,
):
    """Score one saved 50-movie sample and replace only this prompt version's files."""
    movies = load_catalog(catalog)
    template = read_prompt(prompt)
    version = prompt_version(template)
    output_dir = Path(output_dir)
    ids_path = Path(ids_path) if ids_path else output_dir / "pilot_movie_ids.csv"
    movie_ids, reused, stored_seed, stored_n = resolve_pilot_ids(ids_path, movies, n, seed)
    selected = movies_for_ids(movies, movie_ids)
    if complete is None:
        complete = OllamaClient(model, base_url, timeout).complete
    destination = output_directory(output_dir, "pilot", model, template)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index, movie in enumerate(selected, 1):
        _log(verbose, f"pilot {index}/{len(selected)} movie_id={movie['movie_id']}")
        results.append(extract_movie(movie, template, complete))
    summary = export_pilot(destination, selected, results, model, version)
    provenance = destination / "provenance.json"
    write_json(provenance, _provenance(
        "extract-pilot", catalog, prompt, template, model, base_url,
        {
            "pilot_seed": stored_seed,
            "requested_n": stored_n,
            "pilot_movie_ids": str(ids_path),
            "reused_pilot_ids": reused,
            "n_movies": len(selected),
        },
    ))
    summary.update({
        "pilot_movie_ids": str(ids_path),
        "reused_pilot_ids": reused,
        "pilot_seed": stored_seed,
        "requested_n": stored_n,
        "n_movies": len(selected),
        "prompt_version": version,
        "model": model,
        "provenance": str(provenance),
        "output_dir": str(destination),
    })
    return summary


def run_catalog(
    catalog, prompt, output_dir, model="qwen2.5:3b",
    base_url="http://127.0.0.1:11434", timeout=180, retry_failed=False,
    complete=None, verbose=True,
):
    """Extract the full catalog. Completed checkpoints are skipped on the next run."""
    movies = load_catalog(catalog)
    template = read_prompt(prompt)
    version = prompt_version(template)
    destination = output_directory(output_dir, "catalog", model, template)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "checkpoints.jsonl"
    latest = load_checkpoints(checkpoint_path)
    if complete is None:
        complete = OllamaClient(model, base_url, timeout).complete
    skip_statuses = {"ok", "missing_overview"}
    if not retry_failed:
        skip_statuses.add("extraction_failed")
    skipped = 0
    extracted = 0
    try:
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as sink:
            for index, movie in enumerate(movies, 1):
                previous = latest.get(movie["movie_id"])
                if previous and previous.get("status") in skip_statuses:
                    skipped += 1
                    continue
                _log(verbose, f"catalog {index}/{len(movies)} movie_id={movie['movie_id']}")
                result = extract_movie(movie, template, complete)
                record = {
                    "movie_id": result["movie_id"],
                    "tmdb_id": result["tmdb_id"],
                    "title": result["title"],
                    "status": result["status"],
                    "error": result["error"],
                    "model": model,
                    "prompt_version": version,
                    "accepted": result["accepted"],
                    "rejected": result["rejected"],
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                latest[movie["movie_id"]] = record
                extracted += 1
    finally:
        summary = export_catalog(destination, movies, latest, model, version)
        provenance = destination / "provenance.json"
        write_json(provenance, _provenance(
            "extract-catalog", catalog, prompt, template, model, base_url,
            {"retry_failed": retry_failed, "n_catalog_movies": len(movies)},
        ))
        summary.update({
            "checkpoints": str(checkpoint_path),
            "provenance": str(provenance),
            "output_dir": str(destination),
            "prompt_version": version,
            "model": model,
            "n_movies": len(movies),
            "n_skipped": skipped,
            "n_extracted_this_run": extracted,
        })
    return summary

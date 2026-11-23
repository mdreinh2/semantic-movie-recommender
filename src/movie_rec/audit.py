"""Member 5: manual audits of LLM triples and graph-path explanations.

Triple audit (proposal Section V.D):
  * Sample 200 triples from Member 3's semantic_triples.csv (seed 42)
  * Two reviewers mark correct / relevant / ontology_consistent (1 or 0)
  * Report the share judged valid (all three = 1), target >= 80%
  * Cohen's kappa for agreement on each criterion
  * Share of processed movies with at least one accepted triple, target >= 90%

Explanation audit:
  * Sample 30 seed movies ranked by both the metadata-only and LLM-enhanced KG
  * Take each system's rank-1 recommendation and path explanation
  * Reviewers score accuracy and usefulness from 1 to 5 without seeing which
    system produced the item (a separate key file holds that)
  * Report mean scores (target >= 4.0), linear-weighted kappa, and a paired
    Wilcoxon test across seeds
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from .evaluate import _open_text

TRIPLE_CRITERIA = ("correct", "relevant", "ontology_consistent")
EXPLANATION_CRITERIA = ("accuracy", "usefulness")
TRIPLE_TARGET = 0.80
COVERAGE_TARGET = 0.90
EXPLANATION_TARGET = 4.0
PROCESSED_STATUSES = {"ok", "extraction_failed"}
_YES = {"1", "y", "yes", "true", "t"}
_NO = {"0", "n", "no", "false", "f"}


def _read_csv(path):
    with _open_text(path) as source:
        return list(csv.DictReader(source))


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _catalog_lookup(catalog_path):
    return {int(row["movie_id"]): row for row in _read_csv(catalog_path)}


def cohen_kappa(a, b, labels=None, weights=None):
    """Cohen's kappa; weights=None (nominal) or "linear" (ordinal)."""
    if len(a) != len(b) or not a:
        raise ValueError("Kappa needs two equal, non-empty rating lists.")
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    n = len(a)
    observed = [[0.0] * size for _ in range(size)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1.0 / n
    row = [sum(observed[i]) for i in range(size)]
    col = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    if weights is None:
        weight = [[0.0 if i == j else 1.0 for j in range(size)] for i in range(size)]
    elif weights == "linear":
        span = max(size - 1, 1)
        weight = [[abs(i - j) / span for j in range(size)] for i in range(size)]
    else:
        raise ValueError("weights must be None or 'linear'.")
    disagree_observed = sum(weight[i][j] * observed[i][j] for i in range(size) for j in range(size))
    disagree_expected = sum(weight[i][j] * row[i] * col[j] for i in range(size) for j in range(size))
    if disagree_expected == 0:
        return 1.0 if disagree_observed == 0 else 0.0
    return 1.0 - disagree_observed / disagree_expected


def _binary(value, field, audit_id):
    text = (value or "").strip().lower()
    if text in _YES:
        return 1
    if text in _NO:
        return 0
    raise ValueError(f"Item {audit_id}: {field} must be 1 or 0, got {value!r}.")


def _score(value, field, item_id):
    text = (value or "").strip()
    try:
        score = int(float(text))
    except ValueError:
        raise ValueError(f"Item {item_id}: {field} must be 1–5, got {value!r}.") from None
    if not 1 <= score <= 5 or float(text) != score:
        raise ValueError(f"Item {item_id}: {field} must be a whole number 1–5, got {value!r}.")
    return score


def _paired_reviews(review_a, review_b, id_field):
    rows_a = {row[id_field]: row for row in _read_csv(review_a)}
    rows_b = {row[id_field]: row for row in _read_csv(review_b)}
    if set(rows_a) != set(rows_b):
        raise ValueError("The two review sheets list different items.")
    if not rows_a:
        raise ValueError("The review sheets are empty.")
    ids = sorted(rows_a, key=lambda value: int(value) if value.isdigit() else value)
    return ids, rows_a, rows_b


# --------------------------------------------------------------------------
# Triple audit


TRIPLE_SHEET_COLUMNS = (
    "audit_id", "title", "overview", "predicate", "label", "evidence",
    *TRIPLE_CRITERIA, "notes",
)


def sample_triples(triples_csv, catalog_csv, output_dir, n=200, seed=42,
                   reviewers=("A", "B")):
    """Write one blank review sheet per reviewer plus a key file."""
    triples = _read_csv(triples_csv)
    if not triples:
        raise ValueError(f"{triples_csv} contains no triples.")
    if n > len(triples):
        raise ValueError(f"Asked for {n} triples but only {len(triples)} exist.")
    catalog = _catalog_lookup(catalog_csv)
    chosen = random.Random(seed).sample(range(len(triples)), n)
    output_dir = Path(output_dir)
    key_rows, sheet_rows = [], []
    for audit_id, position in enumerate(chosen, 1):
        triple = triples[position]
        movie = catalog.get(int(triple["movie_id"]), {})
        key_rows.append({"audit_id": audit_id, **triple})
        sheet_rows.append({
            "audit_id": audit_id,
            "title": movie.get("title", ""),
            "overview": movie.get("overview", ""),
            "predicate": triple["predicate"],
            "label": triple["normalized_label"],
            "evidence": triple.get("evidence", ""),
            **{criterion: "" for criterion in TRIPLE_CRITERIA},
            "notes": "",
        })
    paths = {"key": _write_csv(output_dir / "triple_audit_key.csv", ["audit_id", *triples[0].keys()], key_rows)}
    for reviewer in reviewers:
        paths[reviewer] = _write_csv(
            output_dir / f"triple_review_{reviewer}.csv", TRIPLE_SHEET_COLUMNS, sheet_rows,
        )
    return {name: str(path) for name, path in paths.items()}


def triple_coverage(status_csv, triples_csv):
    """Share of processed movies with at least one accepted triple."""
    statuses = _read_csv(status_csv)
    processed = {int(row["movie_id"]) for row in statuses if row["status"] in PROCESSED_STATUSES}
    with_triples = {int(row["movie_id"]) for row in _read_csv(triples_csv)}
    counts = Counter(row["status"] for row in statuses)
    covered = len(processed & with_triples)
    return {
        "n_movies": len(statuses),
        "status_counts": dict(sorted(counts.items())),
        "n_processed": len(processed),
        "n_with_triple": covered,
        "coverage": covered / len(processed) if processed else None,
        "target": COVERAGE_TARGET,
        "met_target": bool(processed) and covered / len(processed) >= COVERAGE_TARGET,
    }


def score_triples(review_a, review_b, output_dir=None, status_csv=None, triples_csv=None):
    ids, rows_a, rows_b = _paired_reviews(review_a, review_b, "audit_id")
    marks = {"A": {}, "B": {}}
    for reviewer, rows in (("A", rows_a), ("B", rows_b)):
        for criterion in TRIPLE_CRITERIA:
            marks[reviewer][criterion] = [
                _binary(rows[i].get(criterion), criterion, i) for i in ids
            ]
        marks[reviewer]["valid"] = [
            int(all(marks[reviewer][c][k] for c in TRIPLE_CRITERIA)) for k in range(len(ids))
        ]
    n = len(ids)
    criteria = {}
    for criterion in (*TRIPLE_CRITERIA, "valid"):
        a, b = marks["A"][criterion], marks["B"][criterion]
        rate_a, rate_b = sum(a) / n, sum(b) / n
        criteria[criterion] = {
            "rate_reviewer_a": rate_a,
            "rate_reviewer_b": rate_b,
            "rate_mean": (rate_a + rate_b) / 2,
            "rate_both": sum(x and y for x, y in zip(a, b)) / n,
            "agreement": sum(x == y for x, y in zip(a, b)) / n,
            "kappa": cohen_kappa(a, b, labels=(0, 1)),
        }
    summary = {
        "n_triples": n,
        "criteria": criteria,
        "valid_target": TRIPLE_TARGET,
        "met_valid_target": criteria["valid"]["rate_mean"] >= TRIPLE_TARGET,
        "disagreements": [i for k, i in enumerate(ids) if marks["A"]["valid"][k] != marks["B"]["valid"][k]],
    }
    if status_csv and triples_csv:
        summary["coverage"] = triple_coverage(status_csv, triples_csv)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "triple_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------
# Explanation audit


EXPLANATION_SHEET_COLUMNS = (
    "item_id", "seed_title", "recommended_title", "explanation",
    *EXPLANATION_CRITERIA, "notes",
)


def sample_explanations(system_files, catalog_csv, output_dir, n=30, seed=42,
                        reviewers=("A", "B")):
    """Blind sheets of rank-1 explanations for seeds every system ranks.

    system_files maps a system name to a ranking CSV with an explanation
    column, e.g. {"metadata_kg": ..., "llm_kg": ...}.
    """
    from .final_eval import RankingFileRecommender
    if len(system_files) < 2:
        raise ValueError("Give at least two systems to compare explanations.")
    systems = {name: RankingFileRecommender(path) for name, path in system_files.items()}
    for name, model in systems.items():
        if not model.explanations:
            raise ValueError(f"The {name} rankings have no explanation column.")
    common = sorted(set.intersection(*(set(model.rankings) for model in systems.values())))
    if n > len(common):
        raise ValueError(f"Asked for {n} seeds but only {len(common)} are ranked by every system.")
    rng = random.Random(seed)
    seeds = rng.sample(common, n)
    catalog = _catalog_lookup(catalog_csv)
    items = []
    for seed_id in seeds:
        for name, model in systems.items():
            movie_id = model.rankings[seed_id][0][0]
            items.append({
                "system": name,
                "seed_movie_id": seed_id,
                "movie_id": movie_id,
                "seed_title": catalog.get(seed_id, {}).get("title", ""),
                "recommended_title": catalog.get(movie_id, {}).get("title", ""),
                "explanation": model.explanation(seed_id, movie_id),
            })
    rng.shuffle(items)
    key_rows, sheet_rows = [], []
    for item_id, item in enumerate(items, 1):
        key_rows.append({
            "item_id": item_id, "system": item["system"],
            "seed_movie_id": item["seed_movie_id"], "movie_id": item["movie_id"],
        })
        sheet_rows.append({
            "item_id": item_id,
            "seed_title": item["seed_title"],
            "recommended_title": item["recommended_title"],
            "explanation": item["explanation"],
            **{criterion: "" for criterion in EXPLANATION_CRITERIA},
            "notes": "",
        })
    output_dir = Path(output_dir)
    paths = {"key": _write_csv(
        output_dir / "explanation_audit_key.csv",
        ("item_id", "system", "seed_movie_id", "movie_id"), key_rows,
    )}
    for reviewer in reviewers:
        paths[reviewer] = _write_csv(
            output_dir / f"explanation_review_{reviewer}.csv", EXPLANATION_SHEET_COLUMNS, sheet_rows,
        )
    return {name: str(path) for name, path in paths.items()}


def score_explanations(review_a, review_b, key_csv, output_dir=None):
    from .final_eval import holm_adjust, paired_wilcoxon
    ids, rows_a, rows_b = _paired_reviews(review_a, review_b, "item_id")
    key = {row["item_id"]: row for row in _read_csv(key_csv)}
    if set(ids) - set(key):
        raise ValueError("The key file does not cover every reviewed item.")
    scores = {}
    for item_id in ids:
        for criterion in EXPLANATION_CRITERIA:
            a = _score(rows_a[item_id].get(criterion), criterion, item_id)
            b = _score(rows_b[item_id].get(criterion), criterion, item_id)
            scores[(item_id, criterion)] = (a, b)
    systems = sorted({key[i]["system"] for i in ids})
    per_system = {}
    for system in systems:
        items = [i for i in ids if key[i]["system"] == system]
        block = {"n_items": len(items)}
        for criterion in EXPLANATION_CRITERIA:
            a = [scores[(i, criterion)][0] for i in items]
            b = [scores[(i, criterion)][1] for i in items]
            mean = (sum(a) + sum(b)) / (2 * len(items))
            block[criterion] = {
                "mean_reviewer_a": sum(a) / len(items),
                "mean_reviewer_b": sum(b) / len(items),
                "mean": mean,
                "met_target": mean >= EXPLANATION_TARGET,
            }
        per_system[system] = block
    agreement = {
        criterion: cohen_kappa(
            [scores[(i, criterion)][0] for i in ids],
            [scores[(i, criterion)][1] for i in ids],
            labels=range(1, 6), weights="linear",
        )
        for criterion in EXPLANATION_CRITERIA
    }
    tests = []
    if len(systems) == 2:
        first, second = ("llm_kg", "metadata_kg") if set(systems) == {"llm_kg", "metadata_kg"} else systems
        by_seed = {}
        for i in ids:
            by_seed.setdefault(key[i]["seed_movie_id"], {})[key[i]["system"]] = i
        paired = [pair for pair in by_seed.values() if len(pair) == 2]
        for criterion in EXPLANATION_CRITERIA:
            a = [sum(scores[(pair[first], criterion)]) / 2 for pair in paired]
            b = [sum(scores[(pair[second], criterion)]) / 2 for pair in paired]
            statistic, p_value, n_nonzero = paired_wilcoxon(a, b)
            tests.append({
                "system_a": first, "system_b": second, "criterion": criterion,
                "n_seeds": len(paired), "n_nonzero_differences": n_nonzero,
                "mean_a": sum(a) / len(a), "mean_b": sum(b) / len(b),
                "statistic": statistic, "p_value": p_value,
            })
        for test, adjusted in zip(tests, holm_adjust([t["p_value"] for t in tests])):
            test["p_holm"] = adjusted
    summary = {
        "n_items": len(ids),
        "target_mean": EXPLANATION_TARGET,
        "systems": per_system,
        "weighted_kappa": agreement,
        "tests": tests,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "explanation_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

"""Member 5: final four-system comparison under the shared protocol.

Reuses Member 2's protocol (`load_evaluation_protocol`): the same catalog, seed
queries, held-out labels, and seed-only exclusion. On top of that it adds what
the proposal's evaluation plan asks for:

* Precision@K, Recall@K, NDCG@K, and Hit Rate@K for K = 5, 10, 20
* NDCG@10 as the primary metric
* Paired per-query Wilcoxon signed-rank tests with Holm correction
* A cold-start subset: seed movies with fewer than 10 training ratings
* Coverage: share of queries whose seed has at least one candidate with a
  positive score (Item-KNN returns all-zero scores for cold seeds)

Systems are either saved `.npz` models (TF-IDF, Item-KNN) or precomputed
Top-K ranking files (the knowledge-graph systems). A ranking file is a CSV
(optionally gzipped) with `seed_movie_id`, `rank`, `movie_id`, and optionally
`score` and `explanation`. The rank-catalog command already writes this shape.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluate import _open_text, load_evaluation_protocol

DEFAULT_KS = (5, 10, 20)
METRICS = ("precision", "recall", "ndcg", "hit_rate")
PRIMARY_METRIC = ("ndcg", 10)
COLD_START_MAX_RATINGS = 10
ALPHA = 0.05


class RankingFileRecommender:
    """Serve precomputed Top-K lists, such as Member 4's KG rankings."""

    def __init__(self, path):
        self.path = Path(path)
        self.rankings = {}
        self.explanations = {}
        seen = set()
        with _open_text(self.path) as source:
            reader = csv.DictReader(source)
            missing = {"seed_movie_id", "rank", "movie_id"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{self.path} is missing columns: {', '.join(sorted(missing))}."
                )
            score_column = next(
                (name for name in ("score", "cosine_similarity") if name in reader.fieldnames),
                None,
            )
            has_explanation = "explanation" in reader.fieldnames
            for row in reader:
                seed = int(row["seed_movie_id"])
                rank = int(row["rank"])
                movie = int(row["movie_id"])
                if movie == seed:
                    raise ValueError(f"{self.path}: seed {seed} recommends itself.")
                if (seed, rank) in seen:
                    raise ValueError(f"{self.path}: seed {seed} has rank {rank} twice.")
                seen.add((seed, rank))
                score = float(row[score_column]) if score_column and row[score_column] != "" else 1.0
                self.rankings.setdefault(seed, []).append((rank, movie, score))
                if has_explanation:
                    self.explanations[(seed, movie)] = row["explanation"]
        if not self.rankings:
            raise ValueError(f"{self.path} contains no rankings.")
        for seed, rows in self.rankings.items():
            rows.sort()
            movies = [movie for _, movie, _ in rows]
            if len(set(movies)) != len(movies):
                raise ValueError(f"{self.path}: seed {seed} lists a movie twice.")
            self.rankings[seed] = [(movie, "", score) for _, movie, score in rows]
        # KG rankings only list movies that share an entity with the seed, so
        # some lists are legitimately shorter than K. The longest list shows
        # the K the file was written with.
        self.longest = max(len(rows) for rows in self.rankings.values())

    def list_stats(self, seeds, k):
        """How many of the given seeds have no list or fewer than k items."""
        seeds = set(seeds)
        missing = seeds - set(self.rankings)
        short = {seed for seed in seeds - missing if len(self.rankings[seed]) < k}
        return {
            "n_seeds": len(seeds),
            "n_missing": len(missing),
            "n_short": len(short),
            "mean_length": (
                sum(min(len(self.rankings.get(seed, [])), k) for seed in seeds) / len(seeds)
                if seeds else None
            ),
        }

    def recommend(self, seed_movie_id, k=10, exclude=()):
        if k < 1:
            raise ValueError("k must be positive.")
        return self.rankings.get(int(seed_movie_id), [])[:k]

    def explanation(self, seed_movie_id, movie_id):
        return self.explanations.get((int(seed_movie_id), int(movie_id)), "")


def load_system(path):
    """Load a saved model (.npz) or a ranking file (.csv / .csv.gz)."""
    path = Path(path)
    if path.suffix == ".npz":
        from .itemknn import load_recommender
        return load_recommender(path)
    if path.name.endswith((".csv", ".csv.gz")):
        return RankingFileRecommender(path)
    raise ValueError(f"Unsupported system file {path}. Use .npz or .csv(.gz).")


def parse_system_specs(specs):
    """Turn ["name=path", ...] into an ordered {name: Path}."""
    systems = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        name = name.strip()
        if not separator or not name or not path.strip():
            raise ValueError(f"Use NAME=PATH for --system, got {spec!r}.")
        if name in systems:
            raise ValueError(f"System {name!r} is listed twice.")
        systems[name] = Path(path.strip())
    if not systems:
        raise ValueError("Give at least one --system NAME=PATH.")
    return systems


def training_counts(directory, catalog_ids):
    """Number of training ratings per catalog movie."""
    counts = Counter()
    with _open_text(Path(directory) / "train.csv.gz") as source:
        for row in csv.DictReader(source):
            movie_id = int(row["movie_id"])
            if movie_id in catalog_ids:
                counts[movie_id] += 1
    return counts


def _query_metrics(ranked, relevant, ks, idcg):
    """All metrics at every K from one ranked list (equal to metrics.py)."""
    hit_ranks = [rank for rank, movie in enumerate(ranked, 1) if movie in relevant]
    n_relevant = len(relevant)
    values = np.zeros((len(ks), len(METRICS)))
    for i, k in enumerate(ks):
        hits = [rank for rank in hit_ranks if rank <= k]
        n_hits = len(hits)
        ideal = idcg[min(n_relevant, k)]
        values[i, 0] = n_hits / k
        values[i, 1] = n_hits / n_relevant if n_relevant else 0.0
        values[i, 2] = sum(1.0 / math.log2(rank + 1) for rank in hits) / ideal if ideal else 0.0
        values[i, 3] = 1.0 if n_hits else 0.0
    return values


def score_systems(models, protocol, train_counts, ks=DEFAULT_KS,
                  cold_start_max=COLD_START_MAX_RATINGS, primary=PRIMARY_METRIC):
    """Score every system on identical queries.

    Returns per-system summaries for the full set and the cold-start subset,
    per-query primary-metric arrays for significance testing, and the
    cold-start mask.
    """
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] < 1:
        raise ValueError("Every K must be positive.")
    if primary[1] not in ks:
        raise ValueError(f"The primary cutoff K={primary[1]} must be one of {ks}.")
    max_k = ks[-1]
    queries = protocol["queries"]
    labels = protocol["labels"]
    cold = np.array([train_counts.get(seed, 0) < cold_start_max for _, seed in queries])
    idcg = [0.0]
    for rank in range(1, max_k + 1):
        idcg.append(idcg[-1] + 1.0 / math.log2(rank + 1))
    k_index = ks.index(primary[1])
    m_index = METRICS.index(primary[0])

    results = {}
    for name, model in models.items():
        cache = {}
        totals = {"all": np.zeros((len(ks), len(METRICS))), "cold_start": np.zeros((len(ks), len(METRICS)))}
        covered = {"all": 0, "cold_start": 0}
        primary_values = np.empty(len(queries), dtype=np.float64)
        for position, (user_id, seed_id) in enumerate(queries):
            if seed_id not in cache:
                recs = model.recommend(seed_id, max_k)
                cache[seed_id] = (
                    [movie for movie, _, _ in recs],
                    any(score > 0 for _, _, score in recs),
                )
            ranked, has_signal = cache[seed_id]
            values = _query_metrics(ranked, labels[user_id], ks, idcg)
            totals["all"] += values
            covered["all"] += has_signal
            if cold[position]:
                totals["cold_start"] += values
                covered["cold_start"] += has_signal
            primary_values[position] = values[k_index, m_index]
        subsets = {"all": len(queries), "cold_start": int(cold.sum())}
        summary = {}
        for subset, n in subsets.items():
            rows = {}
            for i, k in enumerate(ks):
                rows[k] = {
                    metric: (float(totals[subset][i, j] / n) if n else None)
                    for j, metric in enumerate(METRICS)
                }
            summary[subset] = {
                "n_queries": n,
                "coverage": covered[subset] / n if n else None,
                "metrics": rows,
            }
        results[name] = {"summary": summary, "primary": primary_values}
    return results, cold


def holm_adjust(p_values):
    """Holm–Bonferroni step-down adjusted p-values, in input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for step, i in enumerate(order):
        running = max(running, min(1.0, (m - step) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test; p = 1 when every pair ties."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    differences = a - b
    n_nonzero = int(np.count_nonzero(differences))
    if n_nonzero == 0:
        return None, 1.0, n_nonzero
    result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue), n_nonzero


def relative_change(new, old):
    if old is None or new is None or old == 0:
        return None
    return 100.0 * (new - old) / old


def significance_tests(results, primary=PRIMARY_METRIC, alpha=ALPHA, target=None):
    """Pairwise tests on the primary metric, Holm-corrected as one family.

    With a target system, the family is target-vs-each-other-system. Without
    one, it is every pair of systems.
    """
    names = list(results)
    if target is not None:
        if target not in results:
            raise ValueError(f"Target system {target!r} was not evaluated.")
        pairs = [(target, other) for other in names if other != target]
    else:
        pairs = list(itertools.combinations(names, 2))
    rows = []
    for first, second in pairs:
        a = results[first]["primary"]
        b = results[second]["primary"]
        statistic, p_value, n_nonzero = paired_wilcoxon(a, b)
        mean_a, mean_b = float(a.mean()), float(b.mean())
        rows.append({
            "system_a": first,
            "system_b": second,
            "metric": f"{primary[0]}@{primary[1]}",
            "n_queries": int(len(a)),
            "n_nonzero_differences": n_nonzero,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference": mean_a - mean_b,
            "relative_change_pct": relative_change(mean_a, mean_b),
            "statistic": statistic,
            "p_value": p_value,
        })
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["p_holm"] = adjusted
        row["significant"] = bool(adjusted < alpha)
    return rows


HYPOTHESES = (
    # (baseline, metric, k, low %, high %) from Section VIII of the proposal.
    ("tfidf", "ndcg", 10, 5.0, 10.0),
    ("metadata_kg", "precision", 10, 3.0, 5.0),
)


def hypothesis_checks(results, target):
    rows = []
    if target not in results:
        return rows
    for baseline, metric, k, low, high in HYPOTHESES:
        if baseline not in results:
            continue
        new = results[target]["summary"]["all"]["metrics"][k][metric]
        old = results[baseline]["summary"]["all"]["metrics"][k][metric]
        change = relative_change(new, old)
        rows.append({
            "comparison": f"{target} vs {baseline}",
            "metric": f"{metric}@{k}",
            "projected_gain_pct": f"{low:g}–{high:g}",
            "observed_change_pct": change,
            "met_projection": bool(change is not None and change >= low),
        })
    return rows


def _fmt(value, digits=5):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _p(value):
    if value is None:
        return "—"
    return "<1e-300" if value == 0 else f"{value:.3g}"


def write_outputs(output_dir, results, tests, hypotheses, protocol, cold, queries,
                  ks, cold_start_max, target, write_query_metrics=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = tuple(sorted(set(ks)))
    paths = {}

    paths["metrics_csv"] = output_dir / "final_metrics.csv"
    with paths["metrics_csv"].open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.writer(target_file, lineterminator="\n")
        writer.writerow(("system", "subset", "k", *METRICS, "n_queries", "coverage"))
        for name, result in results.items():
            for subset, block in result["summary"].items():
                for k in ks:
                    row = block["metrics"][k]
                    writer.writerow((
                        name, subset, k, *(row[metric] for metric in METRICS),
                        block["n_queries"], block["coverage"],
                    ))

    paths["significance_csv"] = output_dir / "significance.csv"
    fields = (
        "system_a", "system_b", "metric", "n_queries", "n_nonzero_differences",
        "mean_a", "mean_b", "mean_difference", "relative_change_pct",
        "statistic", "p_value", "p_holm", "significant",
    )
    with paths["significance_csv"].open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(tests)

    paths["metrics_json"] = output_dir / "final_metrics.json"
    paths["metrics_json"].write_text(json.dumps({
        "ks": list(ks),
        "primary_metric": f"{PRIMARY_METRIC[0]}@{PRIMARY_METRIC[1]}",
        "cold_start_definition": f"seed movie has fewer than {cold_start_max} training ratings",
        "positive_rating_threshold": protocol["positive_rating_threshold"],
        "exclude_seed_only": True,
        "target_system": target,
        "systems": {
            name: {
                subset: {
                    "n_queries": block["n_queries"],
                    "coverage": block["coverage"],
                    "metrics": {str(k): block["metrics"][k] for k in ks},
                }
                for subset, block in result["summary"].items()
            } | ({"list_stats": result["list_stats"]} if "list_stats" in result else {})
            for name, result in results.items()
        },
        "significance": tests,
        "hypotheses": hypotheses,
    }, indent=2) + "\n")

    if write_query_metrics:
        paths["query_csv"] = output_dir / "final_query_primary_metric.csv.gz"
        names = list(results)
        with gzip.open(paths["query_csv"], "wt", encoding="utf-8", newline="") as target_file:
            writer = csv.writer(target_file, lineterminator="\n")
            writer.writerow(("user_id", "seed_movie_id", "cold_start", *names))
            for position, (user_id, seed_id) in enumerate(queries):
                writer.writerow((
                    user_id, seed_id, int(cold[position]),
                    *(f"{results[name]['primary'][position]:.6g}" for name in names),
                ))

    paths["report_md"] = output_dir / "final_report.md"
    paths["report_md"].write_text(
        render_report(results, tests, hypotheses, ks, cold_start_max, target),
        encoding="utf-8",
    )
    return paths


def render_report(results, tests, hypotheses, ks, cold_start_max, target):
    lines = ["# Final evaluation (Member 5)", ""]
    any_result = next(iter(results.values()))["summary"]
    lines.append(
        f"Queries: {any_result['all']['n_queries']:,} overall, "
        f"{any_result['cold_start']['n_queries']:,} cold-start "
        f"(seed has fewer than {cold_start_max} training ratings). "
        f"Primary metric: NDCG@{PRIMARY_METRIC[1]}."
    )
    for subset, title in (("all", "All queries"), ("cold_start", "Cold-start queries")):
        lines += ["", f"## {title}", ""]
        for k in ks:
            lines += [
                f"**K = {k}**", "",
                "| System | Precision | Recall | NDCG | Hit Rate | Coverage |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, result in results.items():
                block = result["summary"][subset]
                row = block["metrics"][k]
                lines.append(
                    f"| {name} | " + " | ".join(_fmt(row[m]) for m in METRICS)
                    + f" | {_fmt(block['coverage'], 3)} |"
                )
            lines.append("")
    listed = {name: result["list_stats"] for name, result in results.items() if "list_stats" in result}
    if listed:
        lines += [
            "## Ranking-list completeness", "",
            f"Graph systems only rank movies that share an entity with the seed, so some "
            f"lists are shorter than K={max(ks)}. Missing positions count as misses.", "",
            "| System | Seeds | No list | Shorter than K | Mean length |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, stats in listed.items():
            lines.append(
                f"| {name} | {stats['n_seeds']:,} | {stats['n_missing']:,} | "
                f"{stats['n_short']:,} | {_fmt(stats['mean_length'], 2)} |"
            )
        lines.append("")
    lines += [
        "## Significance (paired Wilcoxon signed-rank, Holm-corrected)", "",
        "| A | B | Mean A | Mean B | Change % | p | p (Holm) | Significant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tests:
        lines.append(
            f"| {row['system_a']} | {row['system_b']} | {_fmt(row['mean_a'])} | "
            f"{_fmt(row['mean_b'])} | {_fmt(row['relative_change_pct'], 2)} | "
            f"{_p(row['p_value'])} | {_p(row['p_holm'])} | {_fmt(row['significant'])} |"
        )
    if hypotheses:
        lines += [
            "", f"## Projected gains for {target}", "",
            "| Comparison | Metric | Projected % | Observed % | Met |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in hypotheses:
            lines.append(
                f"| {row['comparison']} | {row['metric']} | {row['projected_gain_pct']} | "
                f"{_fmt(row['observed_change_pct'], 2)} | {_fmt(row['met_projection'])} |"
            )
    lines += [
        "", "Queries from the same user share labels, so per-query tests treat "
        "correlated queries as independent. This is a seed-movie protocol, not a "
        "fully personalized evaluation.", "",
    ]
    return "\n".join(lines)


def run_final_evaluation(systems, directory, output_dir, ks=DEFAULT_KS, limit=None,
                         cold_start_max=COLD_START_MAX_RATINGS, target=None,
                         catalog=None, write_query_metrics=False):
    """Load systems, score them, test them, and write every output file."""
    directory = Path(directory)
    catalog = Path(catalog) if catalog else directory / "catalog.csv"
    from .itemknn import file_sha256
    catalog_hash = file_sha256(catalog)
    models = {}
    for name, path in systems.items():
        model = load_system(path)
        stored = getattr(model, "catalog_sha256", None)
        if stored is not None and stored != catalog_hash:
            raise ValueError(f"The {name} model was fitted on a different catalog.")
        if isinstance(model, RankingFileRecommender) and model.longest < max(ks):
            raise ValueError(
                f"{name} has at most {model.longest} recommendations per seed, but "
                f"K={max(ks)} needs {max(ks)}. Rerun its rank-catalog with --k {max(ks)}."
            )
        models[name] = model
    if target is None and "llm_kg" in models:
        target = "llm_kg"
    protocol = load_evaluation_protocol(directory, limit=limit)
    counts = training_counts(directory, protocol["catalog_ids"])
    results, cold = score_systems(models, protocol, counts, ks, cold_start_max)
    query_seeds = {seed for _, seed in protocol["queries"]}
    for name, model in models.items():
        if isinstance(model, RankingFileRecommender):
            results[name]["list_stats"] = model.list_stats(query_seeds, max(ks))
    tests = significance_tests(results, target=target) if len(results) > 1 else []
    hypotheses = hypothesis_checks(results, target) if target else []
    paths = write_outputs(
        output_dir, results, tests, hypotheses, protocol, cold, protocol["queries"],
        ks, cold_start_max, target, write_query_metrics,
    )
    return results, tests, hypotheses, paths

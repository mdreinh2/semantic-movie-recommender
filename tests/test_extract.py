import csv
import json
import os
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from movie_rec.extract import (
    ModelError,
    OllamaClient,
    load_catalog,
    model_dirname,
    normalize_label,
    output_directory,
    parse_triples_payload,
    prompt_version,
    read_prompt,
    render_prompt,
    run_catalog,
    run_pilot,
    select_pilot_movie_ids,
    validate_triples,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "member3_catalog.csv"
PROMPT = ROOT / "prompts" / "extract_triples.txt"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def write_catalog(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("movie_id", "tmdb_id", "title", "overview"))
        writer.writerows(rows)


class ExtractionTests(unittest.TestCase):
    def test_shipped_prompt_exposes_overview_placeholder(self):
        text = PROMPT.read_text(encoding="utf-8")
        self.assertIn("{overview}", text)
        self.assertNotIn("{title}", text)
        self.assertNotIn("{movie_id}", text)
        self.assertNotIn("{tmdb_id}", text)
        for predicate in ("HAS_THEME", "HAS_SETTING", "HAS_CONCEPT"):
            self.assertIn(predicate, text)
        self.assertEqual(prompt_version(text), prompt_version(read_prompt(PROMPT)))

    def test_normalize_label_folds_case_punctuation_and_leading_articles(self):
        self.assertEqual(normalize_label("  The Coming-of-Age! "), "coming of age")
        self.assertEqual(normalize_label("REVENGE"), "revenge")
        self.assertEqual(normalize_label("war of the worlds"), "war of the worlds")
        self.assertEqual(normalize_label("!!!"), "")

    def test_validate_accepts_exact_quotes_and_records_rejection_reasons(self):
        overview = "Two rivals seek revenge after a betrayal in a desert town."
        accepted, rejected = validate_triples(overview, [
            {"predicate": "HAS_THEME", "label": "Revenge", "quote": "revenge"},
            {"predicate": "has_setting", "label": "Desert Town", "quote": "desert town"},
            {"predicate": "HAS_CONCEPT", "label": "betrayal", "quote": "betrayal"},
            {"predicate": "HAS_GENRE", "label": "rivals", "quote": "rivals"},
            {"predicate": "HAS_THEME", "label": "the revenge", "quote": "revenge"},
            {"predicate": "HAS_THEME", "label": "!!!", "quote": "revenge"},
            {"predicate": "HAS_THEME", "label": "towns", "quote": "towns"},
            {"predicate": "HAS_THEME", "label": "inside", "quote": "own"},
            {"predicate": "HAS_THEME", "label": " ", "quote": "revenge"},
            {"predicate": "HAS_CONCEPT", "label": "betrayal", "quote": "  "},
            "not-a-triple",
        ])
        self.assertEqual(
            [(item["predicate"], item["normalized_label"], item["evidence"]) for item in accepted],
            [
                ("HAS_THEME", "revenge", "revenge"),
                ("HAS_SETTING", "desert town", "desert town"),
                ("HAS_CONCEPT", "betrayal", "betrayal"),
            ],
        )
        self.assertEqual(
            [item["reason"] for item in rejected],
            [
                "invalid_predicate", "duplicate", "label_empty_after_normalization",
                "quote_not_in_overview", "quote_not_exact_span", "missing_label",
                "missing_quote", "invalid_item",
            ],
        )
        self.assertEqual(validate_triples(overview, []), ([], []))

    def test_parse_allows_empty_triples_and_json_fences(self):
        self.assertEqual(parse_triples_payload('{"triples": []}'), [])
        fenced = '```json\n{"triples": [{"predicate": "HAS_THEME"}]}\n```'
        self.assertEqual(parse_triples_payload(fenced), [{"predicate": "HAS_THEME"}])
        with self.assertRaisesRegex(ValueError, "invalid_json"):
            parse_triples_payload("not json")
        with self.assertRaisesRegex(ValueError, "missing_triples"):
            parse_triples_payload("{}")
        with self.assertRaisesRegex(ValueError, "triples_not_list"):
            parse_triples_payload('{"triples": {}}')

    def test_select_pilot_is_stable_and_skips_blank_overviews(self):
        movies = [
            {"movie_id": movie_id, "overview": "" if movie_id > 55 else f"Overview {movie_id}"}
            for movie_id in range(1, 61)
        ]
        movies.append({"movie_id": 1, "overview": "duplicate overview"})
        movies.append({"movie_id": 70, "overview": "   "})
        first = select_pilot_movie_ids(movies, 50, 42)
        second = select_pilot_movie_ids(movies, 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(len(set(first)), 50)
        self.assertTrue(set(first).isdisjoint(set(range(56, 61)) | {70}))
        short = [
            {"movie_id": 3, "overview": "one"},
            {"movie_id": 2, "overview": ""},
            {"movie_id": 2, "overview": "again"},
        ]
        self.assertEqual(sorted(select_pilot_movie_ids(short, 50, 42)), [2, 3])

    def test_fixture_pilot_sends_overview_only_and_writes_review(self):
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            if "lost dog" in prompt:
                return json.dumps({"triples": [
                    {"predicate": "HAS_THEME", "label": "Lost Dog", "quote": "lost dog"},
                    {"predicate": "HAS_GENRE", "label": "city", "quote": "city"},
                    {"predicate": "HAS_CONCEPT", "label": "refusal", "quote": "not in the overview"},
                    {"predicate": "has_theme", "label": "the lost dog", "quote": "lost dog"},
                ]})
            if "desert town" in prompt:
                return '{"triples": []}'
            if "coastal laboratory" in prompt:
                return "NOPE"
            raise AssertionError(prompt)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            prompt = output / "prompt.txt"
            write_text(prompt, "Extract from this overview only.\n{overview}\n")
            summary = run_pilot(
                FIXTURE, prompt, output, n=50, seed=42, complete=complete, verbose=False,
            )
            self.assertFalse(summary["reused_pilot_ids"])
            self.assertEqual(summary["pilot_seed"], 42)
            self.assertEqual(summary["n_movies"], 3)
            self.assertEqual(len(prompts), 3)
            sent = "\n".join(prompts)
            self.assertIn("A child searches for a lost dog", sent)
            self.assertNotIn("Alpha Title", sent)
            self.assertNotIn("Beta Title", sent)
            self.assertNotIn("Gamma Title", sent)
            self.assertNotIn("Duplicate Beta", sent)
            self.assertNotIn("must not create another movie", sent)
            self.assertNotIn("101", sent)
            self.assertEqual(render_prompt(read_prompt(prompt), "only text"),
                             "Extract from this overview only.\nonly text\n")

            saved = {int(row["movie_id"]) for row in read_csv(summary["pilot_movie_ids"])}
            self.assertEqual(saved, {1, 2, 5})
            catalog = {movie["movie_id"]: movie for movie in load_catalog(FIXTURE)}
            self.assertEqual(catalog[2]["tmdb_id"], 102)
            self.assertNotIn(999, {movie["tmdb_id"] for movie in catalog.values()})

            review = read_csv(summary["review_csv"])
            self.assertEqual(list(review[0]), [
                "movie_id", "tmdb_id", "title", "overview", "predicate",
                "label", "normalized_label", "quote",
            ])
            self.assertEqual(len(review), 1)
            row = review[0]
            self.assertEqual(row["movie_id"], "1")
            self.assertEqual(row["tmdb_id"], "101")
            self.assertEqual(row["title"], "Alpha Title")
            self.assertIn(",", row["overview"])
            self.assertEqual(row["predicate"], "HAS_THEME")
            self.assertEqual(row["label"], "Lost Dog")
            self.assertEqual(row["normalized_label"], "lost dog")
            self.assertIn(row["quote"], row["overview"])

            reasons = {item["reason"] for item in read_csv(summary["rejections_csv"])}
            self.assertIn("invalid_predicate", reasons)
            self.assertIn("quote_not_in_overview", reasons)
            self.assertIn("duplicate", reasons)
            self.assertIn("invalid_json", reasons)
            statuses = {int(item["movie_id"]): item["status"] for item in read_csv(summary["status_csv"])}
            self.assertEqual(statuses[2], "ok")
            self.assertEqual(statuses[5], "extraction_failed")
            self.assertNotIn(3, statuses)
            self.assertNotIn(4, statuses)

    def test_prompt_revision_reuses_ids_and_keeps_both_results(self):
        def complete(prompt):
            return '{"triples": []}'

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            first_prompt = output / "a.txt"
            second_prompt = output / "b.txt"
            write_text(first_prompt, "A {overview}")
            write_text(second_prompt, "B {overview}")
            first = run_pilot(
                FIXTURE, first_prompt, output, n=50, seed=42, complete=complete, verbose=False,
            )
            saved = Path(first["pilot_movie_ids"]).read_text(encoding="utf-8")
            second = run_pilot(
                FIXTURE, second_prompt, output, n=1, seed=7, complete=complete, verbose=False,
            )
            self.assertTrue(second["reused_pilot_ids"])
            self.assertEqual(second["pilot_seed"], 42)
            self.assertEqual(second["n_movies"], 3)
            self.assertEqual(Path(second["pilot_movie_ids"]).read_text(encoding="utf-8"), saved)
            self.assertNotEqual(first["prompt_version"], second["prompt_version"])
            self.assertNotEqual(first["review_csv"], second["review_csv"])
            self.assertTrue(Path(first["review_csv"]).is_file())
            self.assertTrue(Path(second["review_csv"]).is_file())
            first_ids = {row["movie_id"] for row in read_csv(first["status_csv"])}
            second_ids = {row["movie_id"] for row in read_csv(second["status_csv"])}
            self.assertEqual(first_ids, second_ids)

    def test_catalog_is_resumable_and_records_missing_overviews(self):
        calls = []

        def complete(prompt):
            calls.append(prompt)
            if "ALPHAOVERVIEW" in prompt:
                return json.dumps({"triples": [
                    {"predicate": "HAS_THEME", "label": "loss", "quote": "loss"},
                ]})
            if "BETAOVERVIEW" in prompt:
                raise ModelError("connection refused")
            raise AssertionError(prompt)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            catalog = output / "catalog.csv"
            prompt = output / "prompt.txt"
            write_catalog(catalog, [
                (1, 501, "Alpha", "A story of loss. ALPHAOVERVIEW"),
                (2, 502, "Beta", "A harbor at night. BETAOVERVIEW"),
                (3, 503, "Gamma", ""),
            ])
            write_text(prompt, "{overview}")
            first = run_catalog(catalog, prompt, output, complete=complete, verbose=False)
            self.assertEqual(len(calls), 2)
            self.assertEqual(first["n_skipped"], 0)
            self.assertEqual(first["n_extracted_this_run"], 3)
            self.assertEqual(first["n_ok"], 1)
            self.assertEqual(first["n_failed"], 1)
            self.assertEqual(first["n_missing_overview"], 1)
            self.assertEqual(
                list(read_csv(first["triples_csv"])[0]),
                [
                    "movie_id", "tmdb_id", "predicate", "normalized_label",
                    "evidence", "model", "prompt_version",
                ],
            )
            triple = read_csv(first["triples_csv"])[0]
            self.assertEqual(triple["movie_id"], "1")
            self.assertEqual(triple["tmdb_id"], "501")
            self.assertEqual(triple["predicate"], "HAS_THEME")
            self.assertEqual(triple["normalized_label"], "loss")
            self.assertEqual(triple["evidence"], "loss")
            self.assertEqual(triple["model"], "qwen2.5:3b")
            self.assertEqual(triple["prompt_version"], first["prompt_version"])
            failed_ids = {row["movie_id"] for row in read_csv(first["triples_csv"])}
            self.assertNotIn("2", failed_ids)
            self.assertNotIn("3", failed_ids)
            status = {row["movie_id"]: row["status"] for row in read_csv(first["status_csv"])}
            self.assertEqual(status["3"], "missing_overview")
            self.assertIn("model_error", read_csv(first["status_csv"])[1]["error"])

            calls.clear()
            second = run_catalog(catalog, prompt, output, complete=complete, verbose=False)
            self.assertEqual(calls, [])
            self.assertEqual(second["n_skipped"], 3)
            self.assertEqual(second["n_extracted_this_run"], 0)
            self.assertEqual(len(read_csv(second["triples_csv"])), 1)

    def test_retry_failed_replaces_failed_checkpoint(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            catalog = output / "catalog.csv"
            prompt = output / "prompt.txt"
            write_catalog(catalog, [
                (1, 501, "Alpha", "A story of loss. ALPHAOVERVIEW"),
                (2, 502, "Beta", "A harbor at night. BETAOVERVIEW"),
            ])
            write_text(prompt, "{overview}")

            def fail_beta(prompt_text):
                if "BETAOVERVIEW" in prompt_text:
                    raise ModelError("timed out")
                return json.dumps({"triples": [
                    {"predicate": "HAS_THEME", "label": "loss", "quote": "loss"},
                ]})

            run_catalog(catalog, prompt, output, complete=fail_beta, verbose=False)
            calls = []

            def repair_beta(prompt_text):
                calls.append(prompt_text)
                self.assertIn("BETAOVERVIEW", prompt_text)
                return json.dumps({"triples": [
                    {"predicate": "HAS_SETTING", "label": "harbor", "quote": "harbor"},
                ]})

            repaired = run_catalog(
                catalog, prompt, output, complete=repair_beta, retry_failed=True, verbose=False,
            )
            self.assertEqual(len(calls), 1)
            labels = {row["normalized_label"]: row for row in read_csv(repaired["triples_csv"])}
            self.assertEqual(set(labels), {"loss", "harbor"})
            self.assertEqual(labels["harbor"]["tmdb_id"], "502")
            self.assertEqual(labels["harbor"]["evidence"], "harbor")
            statuses = {row["movie_id"]: row["status"] for row in read_csv(repaired["status_csv"])}
            self.assertEqual(statuses["2"], "ok")

    def test_partial_checkpoint_line_is_resumed(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            catalog = output / "catalog.csv"
            prompt = output / "prompt.txt"
            write_catalog(catalog, [
                (1, 101, "Alpha", "A story of loss."),
                (2, 202, "Beta", "A harbor at night. BETAOVERVIEW"),
            ])
            write_text(prompt, "{overview}")
            template = read_prompt(prompt)
            destination = output_directory(output, "catalog", "qwen2.5:3b", template)
            destination.mkdir(parents=True)
            record = {
                "movie_id": 1,
                "tmdb_id": 555,
                "title": "Wrong",
                "status": "ok",
                "error": "",
                "model": "qwen2.5:3b",
                "prompt_version": prompt_version(template),
                "accepted": [{
                    "predicate": "HAS_THEME",
                    "normalized_label": "loss",
                    "raw_label": "loss",
                    "evidence": "loss",
                }],
                "rejected": [],
            }
            checkpoint = destination / "checkpoints.jsonl"
            checkpoint.write_text(json.dumps(record) + "\n{bad", encoding="utf-8")
            calls = []

            def complete(prompt_text):
                calls.append(prompt_text)
                self.assertIn("BETAOVERVIEW", prompt_text)
                self.assertNotIn("loss", prompt_text)
                return json.dumps({"triples": [
                    {"predicate": "HAS_SETTING", "label": "harbor", "quote": "harbor"},
                ]})

            summary = run_catalog(catalog, prompt, output, complete=complete, verbose=False)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("{bad", checkpoint.read_text(encoding="utf-8"))
            rows = {row["movie_id"]: row for row in read_csv(summary["triples_csv"])}
            self.assertEqual(rows["1"]["tmdb_id"], "101")
            self.assertEqual(rows["1"]["normalized_label"], "loss")
            self.assertEqual(rows["2"]["tmdb_id"], "202")
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                json.loads(line)

    def test_ollama_client_posts_only_the_prompt(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"message": {"content": "{\"triples\": []}"}}).encode()

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode())
            return Response()

        with mock.patch("movie_rec.extract.urllib.request.urlopen", fake_urlopen):
            text = OllamaClient("qwen2.5:3b", timeout=12).complete("ONLYTHISPROMPT")
        self.assertEqual(text, "{\"triples\": []}")
        self.assertTrue(captured["url"].endswith("/api/chat"))
        self.assertEqual(captured["timeout"], 12)
        body = captured["body"]
        self.assertEqual(body["model"], "qwen2.5:3b")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["options"], {"temperature": 0})
        self.assertEqual(body["messages"], [{"role": "user", "content": "ONLYTHISPROMPT"}])
        self.assertEqual(model_dirname("qwen2.5:3b"), "qwen2.5-3b")

    def test_ollama_connection_error_is_a_model_error(self):
        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("connection refused")

        with mock.patch("movie_rec.extract.urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(ModelError) as caught:
                OllamaClient("qwen2.5:3b").complete("overview")
        self.assertIn("connection refused", str(caught.exception))

    def test_cli_tells_user_to_get_member1_catalog(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "catalog.csv"
            result = subprocess.run(
                [sys.executable, "-m", "movie_rec.cli", "extract-pilot", "--catalog", str(missing)],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("catalog.csv", result.stderr)
        self.assertIn("Member 1", result.stderr)
        self.assertIn("data/processed/catalog.csv", result.stderr)
        help_result = subprocess.run(
            [sys.executable, "-m", "movie_rec.cli", "-h"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        )
        self.assertIn("extract-pilot", help_result.stdout)
        self.assertIn("extract-catalog", help_result.stdout)


if __name__ == "__main__":
    unittest.main()

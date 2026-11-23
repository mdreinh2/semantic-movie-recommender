"""Local web UI: pick a seed movie, compare the four systems, see why.

Run from the project folder after the pipeline and run_member5.py:

    python scripts/ui_server.py

then open http://localhost:8000. Uses only the standard library plus this
project's own loaders, and serves only on this computer (127.0.0.1).
"""

import argparse
import csv
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_rec.final_eval import load_system  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SYSTEMS = (
    ("metadata_kg", "Metadata KG", "member4/metadata_kg_rankings.csv"),
    ("llm_kg", "LLM-enhanced KG", "member4/llm_kg_rankings.csv"),
    ("itemknn", "Item-KNN", "itemknn_model.npz"),
    ("tfidf", "TF-IDF", "tfidf_model.npz"),
)
RELATIONS = {
    "HAS_GENRE": ("Genre", False), "HAS_KEYWORD": ("Keyword", False),
    "HAS_ACTOR": ("Actor", False), "HAS_DIRECTOR": ("Director", False),
    "HAS_THEME": ("Theme", True), "HAS_SETTING": ("Setting", True),
    "HAS_CONCEPT": ("Concept", True),
}
PATH = re.compile(r"-\[(HAS_[A-Z]+)\]-> \((.+?)\) <-\[\1\]-")
NOTES = {
    "itemknn": "Users who rated the seed highly also rated this highly.",
    "tfidf": "Similar words in the overview, genres and keywords.",
}


def load_catalog(path):
    movies = {}
    with open(path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                genres = json.loads(row.get("genres") or "[]")
            except ValueError:
                genres = []
            movies[int(row["movie_id"])] = {
                "id": int(row["movie_id"]),
                "title": row["title"],
                "year": (row.get("release_date") or "")[:4],
                "genres": genres[:4],
                "train_count": int(row.get("train_count") or 0),
            }
    return movies


def parse_paths(text):
    chips = []
    for relation, entity in PATH.findall(text or ""):
        label, from_llm = RELATIONS.get(relation, (relation.title(), False))
        chips.append({"kind": label, "entity": entity, "llm": from_llm})
    return chips


class App:
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.movies = load_catalog(data_dir / "catalog.csv")
        self.systems = {}
        for key, label, name in SYSTEMS:
            path = data_dir / name
            for candidate in (path, path.with_name(path.name + ".gz")):
                if candidate.exists():
                    print(f"Loading {label} from {candidate.name} ...", flush=True)
                    self.systems[key] = (label, load_system(candidate))
                    break
            else:
                print(f"Skipping {label}: {path} not found", flush=True)
        if not self.systems:
            raise SystemExit("No systems found. Run the pipeline and run_member5.py first.")

    def search(self, query, limit=12):
        query = query.strip().lower()
        if not query:
            return []
        hits = [m for m in self.movies.values() if query in m["title"].lower()]
        hits.sort(key=lambda m: (not m["title"].lower().startswith(query), -m["train_count"]))
        return hits[:limit]

    def recommend(self, seed, system, k=10):
        if seed not in self.movies:
            raise ValueError("That movie is not in the 5,000-movie catalog.")
        if system not in self.systems:
            raise ValueError("Unknown system.")
        _, model = self.systems[system]
        recs = model.recommend(seed, k)
        items = []
        for rank, (movie_id, _, score) in enumerate(recs, 1):
            movie = self.movies.get(movie_id, {"title": str(movie_id), "year": ""})
            explain = getattr(model, "explanation", None)
            items.append({
                "rank": rank, "id": movie_id, "title": movie["title"], "year": movie["year"],
                "score": round(float(score), 4),
                "chips": parse_paths(explain(seed, movie_id)) if explain else [],
            })
        no_signal = bool(items) and all(item["score"] <= 0 for item in items)
        return {
            "seed": self.movies[seed], "system": system,
            "items": [] if no_signal else items,
            "note": NOTES.get(system, "Shared graph entities connect each pick to the seed."),
            "empty_reason": (
                "No one in the training data co-rated this movie, so Item-KNN has no answer (cold start)."
                if no_signal or (not items and system == "itemknn")
                else "This graph has no movie sharing an entity with the seed." if not items else ""
            ),
        }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Group 13 · Explainable Movie Recommender</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--maroon:#8C1D40;--navy:#1D3557;--cream:#F3E9D2;--bg:#FBFAF7;--ink:#2B2B2B;--muted:#6B6460;--line:#E3DDD0;--llm:#DCE8F7}
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--maroon);color:var(--bg);padding:18px 32px}header h1{margin:0;font-size:24px}
header p{margin:4px 0 0;font-size:14px;opacity:.9}
main{max-width:1100px;margin:24px auto;padding:0 24px}
.search{position:relative}
#q{width:100%;font-size:18px;padding:12px 16px;border:2px solid var(--navy);border-radius:10px;background:#fff}
#sugg{position:absolute;left:0;right:0;top:52px;background:#fff;border:1px solid var(--line);border-radius:10px;
box-shadow:0 8px 24px rgba(0,0,0,.08);z-index:5;max-height:360px;overflow:auto;display:none}
#sugg button{display:block;width:100%;text-align:left;padding:10px 16px;border:0;background:none;font-size:16px;cursor:pointer}
#sugg button:hover,#sugg button:focus{background:var(--cream)}
.seed{margin:18px 0;padding:14px 18px;background:#fff;border:1px solid var(--line);border-radius:12px;display:none}
.seed h2{margin:0 0 4px;font-size:22px}.seed p{margin:0;color:var(--muted);font-size:15px}
.cold{display:inline-block;margin-left:8px;padding:2px 8px;border-radius:6px;background:#FFE7C2;color:#6A3B00;font-size:13px;font-weight:700}
.tabs{display:flex;gap:8px;margin:8px 0 14px;flex-wrap:wrap}
.tabs button{font-size:15px;padding:8px 16px;border-radius:999px;border:0;background:#EFEAE0;color:#3A2E2A;cursor:pointer}
.tabs button[aria-pressed=true]{background:var(--maroon);color:var(--bg);font-weight:700}
.note{color:var(--muted);font-size:14px;margin:0 0 10px}
ol{list-style:none;margin:0;padding:0}
li.rec{display:flex;gap:16px;padding:12px 0;border-top:1px solid var(--line)}
.rank{font-weight:700;color:var(--maroon);font-size:20px;width:28px}
.t{font-weight:700;font-size:18px}.y,.s{color:var(--muted);font-weight:400;font-size:14px;margin-left:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:13px;padding:3px 10px;border-radius:6px;background:var(--cream);color:#3A2E2A}
.chip.llm{background:var(--llm);color:#16324F}.chip b{font-weight:700}
.empty{padding:16px;background:#FFF4D1;border-radius:10px;color:#5A4300}
footer{color:var(--muted);font-size:13px;margin:28px 0}
</style></head><body>
<header><h1>Explainable Movie Recommender</h1>
<p>CSE 573 · Group 13 · TF-IDF, Item-KNN and knowledge-graph recommendations with graph-path explanations</p></header>
<main>
<div class="search"><input id="q" placeholder="Search a seed movie, e.g. Toy Story" autocomplete="off" aria-label="Search a seed movie">
<div id="sugg" role="listbox"></div></div>
<div class="seed" id="seed"></div>
<div class="tabs" id="tabs" role="toolbar" aria-label="Recommender"></div>
<p class="note" id="note"></p>
<ol id="list"></ol>
<footer>Catalog: 5,000 MovieLens 20M movies. Blue tags come from Qwen2.5-3B LLM triples; tan tags from TMDB metadata.</footer>
</main>
<script>
const SYSTEMS = __SYSTEMS__;
let seed = null, system = SYSTEMS[0][0], timer = null;
const $ = id => document.getElementById(id);
function el(tag, cls, text){const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e;}
function tabs(){const t=$("tabs"); t.textContent="";
  for(const [key,label] of SYSTEMS){const b=el("button",null,label); b.setAttribute("aria-pressed", key===system);
    b.onclick=()=>{system=key; tabs(); load();}; t.appendChild(b);}}
$("q").addEventListener("input", e=>{clearTimeout(timer); timer=setTimeout(()=>suggest(e.target.value),150);});
async function suggest(q){const box=$("sugg"); box.textContent="";
  if(!q.trim()){box.style.display="none"; return;}
  const hits=await (await fetch("/api/search?q="+encodeURIComponent(q))).json();
  for(const m of hits){const b=el("button",null,m.title+(m.year?" ("+m.year+")":""));
    b.onclick=()=>{seed=m.id; box.style.display="none"; $("q").value=m.title; load();}; box.appendChild(b);}
  box.style.display=hits.length?"block":"none";}
async function load(){if(seed===null) return;
  const r=await fetch(`/api/recommend?seed=${seed}&system=${system}`); const d=await r.json();
  if(d.error){$("list").textContent=""; $("list").appendChild(el("li","empty",d.error)); return;}
  const s=$("seed"); s.style.display="block"; s.textContent="";
  const h=el("h2",null,d.seed.title+(d.seed.year?" ("+d.seed.year+")":""));
  if(d.seed.train_count<10) h.appendChild(el("span","cold","Cold start: "+d.seed.train_count+" training ratings"));
  s.appendChild(h); s.appendChild(el("p",null,(d.seed.genres||[]).join(" · ")+"  |  "+d.seed.train_count.toLocaleString()+" training ratings"));
  $("note").textContent=d.note; const list=$("list"); list.textContent="";
  if(!d.items.length){list.appendChild(el("li","empty",d.empty_reason)); return;}
  for(const it of d.items){const li=el("li","rec"); li.appendChild(el("div","rank",it.rank));
    const body=el("div"); const t=el("div","t",it.title); t.appendChild(el("span","y",it.year?"("+it.year+")":""));
    t.appendChild(el("span","s","score "+it.score)); body.appendChild(t);
    if(it.chips.length){const c=el("div","chips");
      for(const ch of it.chips){const x=el("span","chip"+(ch.llm?" llm":"")); x.appendChild(el("b",null,ch.kind+(ch.llm?" (LLM)":"")+": "));
        x.appendChild(document.createTextNode(ch.entity)); c.appendChild(x);} body.appendChild(c);}
    li.appendChild(body); list.appendChild(li);}}
tabs();
</script></body></html>"""


def make_handler(app):
    page = PAGE.replace(
        "__SYSTEMS__", json.dumps([[key, label] for key, (label, _) in app.systems.items()]),
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, kind="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    return self._send(page, "text/html")
                if url.path == "/api/search":
                    return self._send(json.dumps(app.search(query.get("q", [""])[0])).encode())
                if url.path == "/api/recommend":
                    data = app.recommend(int(query["seed"][0]), query["system"][0])
                    return self._send(json.dumps(data).encode())
                return self._send(b'{"error":"not found"}', status=404)
            except (KeyError, ValueError) as error:
                return self._send(json.dumps({"error": str(error), "items": []}).encode(), status=400)

        def log_message(self, *args):
            return None

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    app = App(args.data_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Open http://localhost:{args.port} in your browser (Ctrl+C to stop).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

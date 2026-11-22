import csv
import json
from collections import defaultdict
from pathlib import Path
METADATA_RELATIONS={"genres":"HAS_GENRE","keywords":"HAS_KEYWORD","cast":"HAS_ACTOR","directors":"HAS_DIRECTOR"}
LLM_RELATIONS={"theme":"HAS_THEME","setting":"HAS_SETTING","concept":"HAS_CONCEPT"}
NODE_LABELS={"HAS_GENRE":"Genre","HAS_KEYWORD":"Keyword","HAS_ACTOR":"Person","HAS_DIRECTOR":"Person","HAS_THEME":"Theme","HAS_SETTING":"Setting","HAS_CONCEPT":"Concept"}
def normalize_entity(text):
    return " ".join(str(text).split()).lower()
class MovieGraph:
    def __init__(self):
        self.titles={}
        self.edges=defaultdict(dict)
    def add_movie(self, movie_id, title):
        self.titles[movie_id]=title
    def add_edge(self, movie_id, relation, entity, source, confidence=1.0):
        key=(relation, entity)
        existing=self.edges[movie_id].get(key)
        if existing is None or confidence>existing[1]:
            self.edges[movie_id][key]=(source, confidence)
    def edge_rows(self):
        for movie_id in sorted(self.edges):
            for (relation, entity), (source, confidence) in sorted(self.edges[movie_id].items()):
                yield {"movie_id":movie_id, "relation":relation, "entity":entity, "source":source, "confidence":confidence}
    def summary(self):
        counts=defaultdict(int)
        entities=defaultdict(set)
        for row in self.edge_rows():
            counts[row["relation"]]+=1
            entities[row["relation"]].add(row["entity"])
        return {
            "movies":len(self.titles),
            "edges_by_relation":dict(sorted(counts.items())),
            "entities_by_relation":{relation:len(values) for relation, values in sorted(entities.items())},
        }
def read_triples(triples_path):
    path=Path(triples_path)
    if path.suffix==".jsonl":
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)
def add_metadata(graph, catalog_path, cast_limit):
    with open(catalog_path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            movie_id=int(row["movie_id"])
            graph.add_movie(movie_id, row["title"])
            for column, relation in METADATA_RELATIONS.items():
                values=json.loads(row[column])
                if column=="cast":
                    values=values[:cast_limit]
                for value in values:
                    entity=normalize_entity(value)
                    if entity:
                        graph.add_edge(movie_id, relation, entity, "metadata")
def add_triples(graph, triples_path, min_confidence):
    skipped=defaultdict(int)
    for triple in read_triples(triples_path):
        relation=LLM_RELATIONS.get(normalize_entity(triple.get("relation", "")))
        try:
            movie_id=int(triple["movie_id"])
            confidence=float(triple.get("confidence") or 1.0)
        except (KeyError, ValueError, TypeError):
            skipped["malformed"]+=1
            continue
        entity=normalize_entity(triple.get("entity", ""))
        if relation is None:
            skipped["unknown_relation"]+=1
        elif movie_id not in graph.titles:
            skipped["movie_outside_catalog"]+=1
        elif not entity:
            skipped["empty_entity"]+=1
        elif confidence<min_confidence:
            skipped["below_min_confidence"]+=1
        else:
            graph.add_edge(movie_id, relation, entity, "llm", confidence)
    return dict(skipped)
def build_graph(catalog_path, triples_path=None, cast_limit=5, min_confidence=0.0):
    graph=MovieGraph()
    add_metadata(graph, catalog_path, cast_limit)
    graph.skipped_triples=add_triples(graph, triples_path, min_confidence) if triples_path else {}
    return graph
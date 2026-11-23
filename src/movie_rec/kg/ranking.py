import math
from dataclasses import dataclass, field
DEFAULT_RELATION_WEIGHTS={
    "HAS_GENRE":1.0,
    "HAS_KEYWORD":1.5,
    "HAS_ACTOR":0.5,
    "HAS_DIRECTOR":1.5,
    "HAS_THEME":2.0,
    "HAS_SETTING":1.5,
    "HAS_CONCEPT":2.0,
}
@dataclass
class Evidence:
    relation: str
    entity: str
    contribution: float
    def path(self, seed_title, candidate_title):
        return f"({seed_title}) -[{self.relation}]-> ({self.entity}) <-[{self.relation}]- ({candidate_title})"
@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    evidence: list=field(default_factory=list)
    def explanation_paths(self, seed_title, limit=3):
        return [item.path(seed_title, self.title) for item in self.evidence[:limit]]
def rank_candidates(rows, total_movies, k=10, relation_weights=None):
    if k<1:
        raise ValueError("k must be positive.")
    weights=relation_weights or DEFAULT_RELATION_WEIGHTS
    grouped={}
    for row in rows:
        idf=math.log((1+total_movies)/(1+row["entity_movie_count"]))+1.0
        contribution=weights.get(row["relation"], 1.0)*idf*min(row["seed_confidence"], row["candidate_confidence"])
        entry=grouped.setdefault(row["candidate_id"], [row["candidate_title"], [], row.get("candidate_degree", 1)])
        entry[1].append(Evidence(row["relation"], row["entity"], contribution))
    recommendations=[]
    for movie_id, (title, evidence, degree) in grouped.items():
        evidence.sort(key=lambda item: (-item.contribution, item.relation, item.entity))
        score=sum(item.contribution for item in evidence)/math.sqrt(max(degree, 1))
        recommendations.append(Recommendation(movie_id, title, score, evidence))
    recommendations.sort(key=lambda item: (-item.score, item.movie_id))
    return recommendations[:k]
class KnowledgeGraphRecommender:
    def __init__(self, store, sources, relation_weights=None):
        self.store=store
        self.sources=tuple(sources)
        self.relation_weights=relation_weights
    def recommend(self, seed_movie_id, k=10):
        if not self.store.has_movie(seed_movie_id):
            raise ValueError(f"MovieLens ID {seed_movie_id} is outside the shared catalog.")
        rows=self.store.shared_entity_rows(seed_movie_id, self.sources)
        return rank_candidates(rows, self.store.movie_count(), k, self.relation_weights)
    def seed_title(self, seed_movie_id):
        return self.store.title(seed_movie_id)
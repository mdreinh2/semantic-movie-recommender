import csv
import json
import tempfile
import unittest
from pathlib import Path
from movie_rec.kg.graph_data import build_graph
from movie_rec.kg.memory_store import MemoryGraphStore
from movie_rec.kg.ranking import KnowledgeGraphRecommender, rank_candidates
COLUMNS=("movie_id", "tmdb_id", "title", "overview", "release_date", "genres", "keywords", "cast", "directors", "train_count", "popularity_band")
MOVIES=[
    (1, "Toy Story", ["Animation", "Family"], ["toy", "friendship"], ["Tom Hanks", "Tim Allen"], ["John Lasseter"]),
    (2, "Toy Story 2", ["Animation", "Family"], ["toy", "rescue"], ["Tom Hanks", "Tim Allen"], ["John Lasseter"]),
    (3, "Heist Night", ["Crime"], ["heist"], ["Tom Hanks"], ["Someone Else"]),
    (4, "Small Soldiers", ["Action", "Family"], ["toy", "war"], ["Kirsten Dunst"], ["Joe Dante"]),
    (5, "Lonely Island", ["Drama"], ["survival"], ["Nobody"], ["Another"]),
]
TRIPLES=[
    (1, "theme", "Friendship  and Loyalty", 0.9),
    (5, "theme", "friendship and loyalty", 0.8),
    (5, "setting", "Desert Island", 0.7),
    (1, "concept", "toys come alive", 0.2),
    (4, "concept", "Toys come alive", 0.95),
    (1, "mood", "cozy", 1.0),
    (99, "theme", "outside catalog", 1.0),
]
def write_fixtures(directory):
    catalog=Path(directory)/"catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as target:
        writer=csv.writer(target)
        writer.writerow(COLUMNS)
        for movie_id, title, genres, keywords, cast, directors in MOVIES:
            writer.writerow((movie_id, movie_id, title, "", "", json.dumps(genres), json.dumps(keywords), json.dumps(cast), json.dumps(directors), 0, "0"))
    triples=Path(directory)/"triples.csv"
    with triples.open("w", encoding="utf-8", newline="") as target:
        writer=csv.writer(target)
        writer.writerow(("movie_id", "relation", "entity", "confidence"))
        writer.writerows(TRIPLES)
    return catalog, triples
class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog, self.triples=write_fixtures(self.temporary.name)
        self.graph=build_graph(self.catalog, self.triples)
        self.store=MemoryGraphStore(self.graph)
        self.metadata=KnowledgeGraphRecommender(self.store, ("metadata",))
        self.enhanced=KnowledgeGraphRecommender(self.store, ("metadata", "llm"))
    def test_triples_are_normalized_and_filtered(self):
        self.assertEqual(self.graph.skipped_triples, {"unknown_relation":1, "movie_outside_catalog":1})
        self.assertIn(("HAS_THEME", "friendship and loyalty"), self.graph.edges[1])
        self.assertIn(("HAS_THEME", "friendship and loyalty"), self.graph.edges[5])
    def test_min_confidence_drops_weak_triples(self):
        strict=build_graph(self.catalog, self.triples, min_confidence=0.5)
        self.assertNotIn(("HAS_CONCEPT", "toys come alive"), strict.edges[1])
        self.assertEqual(strict.skipped_triples["below_min_confidence"], 1)
    def test_metadata_mode_ignores_llm_edges(self):
        ranked=self.metadata.recommend(1, 10)
        self.assertEqual(ranked[0].movie_id, 2)
        self.assertNotIn(5, [item.movie_id for item in ranked])
        for item in ranked:
            self.assertFalse({evidence.relation for evidence in item.evidence} & {"HAS_THEME", "HAS_SETTING", "HAS_CONCEPT"})
    def test_enhanced_mode_reaches_movies_metadata_cannot(self):
        ranked=self.enhanced.recommend(1, 10)
        self.assertIn(5, [item.movie_id for item in ranked])
        island=next(item for item in ranked if item.movie_id==5)
        self.assertEqual([evidence.relation for evidence in island.evidence], ["HAS_THEME"])
    def test_low_confidence_edge_is_discounted(self):
        ranked=self.enhanced.recommend(1, 10)
        soldiers=next(item for item in ranked if item.movie_id==4)
        concept=next(evidence for evidence in soldiers.evidence if evidence.relation=="HAS_CONCEPT")
        keyword=next(evidence for evidence in soldiers.evidence if evidence.relation=="HAS_KEYWORD")
        self.assertLess(concept.contribution, keyword.contribution*2)
    def test_rarer_shared_entities_score_higher(self):
        rows=[
            {"candidate_id":2, "candidate_title":"B", "relation":"HAS_KEYWORD", "entity":"rare", "entity_movie_count":2, "seed_confidence":1.0, "candidate_confidence":1.0},
            {"candidate_id":3, "candidate_title":"C", "relation":"HAS_KEYWORD", "entity":"common", "entity_movie_count":50, "seed_confidence":1.0, "candidate_confidence":1.0},
        ]
        self.assertEqual([item.movie_id for item in rank_candidates(rows, 100)], [2, 3])
    def test_ties_break_by_movie_id_and_seed_is_excluded(self):
        rows=[
            {"candidate_id":movie_id, "candidate_title":str(movie_id), "relation":"HAS_GENRE", "entity":"x", "entity_movie_count":3, "seed_confidence":1.0, "candidate_confidence":1.0}
            for movie_id in (9, 4, 7)
        ]
        self.assertEqual([item.movie_id for item in rank_candidates(rows, 10)], [4, 7, 9])
        self.assertNotIn(1, [item.movie_id for item in self.enhanced.recommend(1, 10)])
    def test_explanation_paths_name_both_movies(self):
        top=self.metadata.recommend(1, 1)[0]
        path=top.explanation_paths("Toy Story", 1)[0]
        self.assertTrue(path.startswith("(Toy Story) -["))
        self.assertTrue(path.endswith("(Toy Story 2)"))
    def test_unknown_seed_and_bad_k(self):
        with self.assertRaises(ValueError):
            self.metadata.recommend(99, 10)
        with self.assertRaises(ValueError):
            self.metadata.recommend(1, 0)
    def test_cast_limit_caps_actors(self):
        limited=build_graph(self.catalog, cast_limit=1)
        actors=[key for key in limited.edges[1] if key[0]=="HAS_ACTOR"]
        self.assertEqual(actors, [("HAS_ACTOR", "tom hanks")])
if __name__=="__main__":
    unittest.main()
from .graph_data import NODE_LABELS
BATCH_SIZE=5000
SHARED_ENTITY_QUERY="""
MATCH (seed:Movie {movie_id:$seed})-[seed_edge]->(entity)<-[candidate_edge]-(candidate:Movie)
WHERE candidate.movie_id<>$seed
AND type(seed_edge)=type(candidate_edge)
AND seed_edge.source IN $sources
AND candidate_edge.source IN $sources
WITH candidate, entity, type(seed_edge) AS relation, seed_edge, candidate_edge
RETURN candidate.movie_id AS candidate_id,
candidate.title AS candidate_title,
relation,
entity.name AS entity,
COUNT { (entity)<-[member_edge]-(:Movie) WHERE type(member_edge)=relation AND member_edge.source IN $sources } AS entity_movie_count,
COUNT { (candidate)-[degree_edge]->() WHERE degree_edge.source IN $sources } AS candidate_degree,
seed_edge.confidence AS seed_confidence,
candidate_edge.confidence AS candidate_confidence
"""
def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start+size]
class Neo4jGraphStore:
    def __init__(self, uri, user, password, database="neo4j"):
        from neo4j import GraphDatabase
        self.driver=GraphDatabase.driver(uri, auth=(user, password))
        self.database=database
    def close(self):
        self.driver.close()
    def run(self, query, **parameters):
        records, _, _=self.driver.execute_query(query, parameters, database_=self.database)
        return [record.data() for record in records]
    def reset(self):
        self.run("MATCH (n) DETACH DELETE n")
    def create_constraints(self):
        self.run("CREATE CONSTRAINT movie_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.movie_id IS UNIQUE")
        for label in sorted(set(NODE_LABELS.values())):
            self.run(f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS FOR (e:{label}) REQUIRE e.name IS UNIQUE")
    def load(self, graph):
        self.create_constraints()
        movies=[{"movie_id":movie_id, "title":title} for movie_id, title in sorted(graph.titles.items())]
        for batch in chunks(movies, BATCH_SIZE):
            self.run("UNWIND $rows AS row MERGE (m:Movie {movie_id:row.movie_id}) SET m.title=row.title", rows=batch)
        by_relation={}
        for row in graph.edge_rows():
            by_relation.setdefault(row["relation"], []).append(row)
        for relation, rows in by_relation.items():
            label=NODE_LABELS[relation]
            query=(
                "UNWIND $rows AS row "
                "MATCH (m:Movie {movie_id:row.movie_id}) "
                f"MERGE (e:{label} {{name:row.entity}}) "
                f"MERGE (m)-[r:{relation}]->(e) "
                "SET r.source=row.source, r.confidence=row.confidence"
            )
            for batch in chunks(rows, BATCH_SIZE):
                self.run(query, rows=batch)
        return self.counts()
    def counts(self):
        nodes=self.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS total ORDER BY label")
        edges=self.run("MATCH ()-[r]->() RETURN type(r) AS relation, r.source AS source, count(*) AS total ORDER BY relation, source")
        return {"nodes":nodes, "edges":edges}
    def has_movie(self, movie_id):
        return bool(self.run("MATCH (m:Movie {movie_id:$id}) RETURN m.movie_id AS id", id=movie_id))
    def movie_count(self):
        return self.run("MATCH (m:Movie) RETURN count(m) AS total")[0]["total"]
    def title(self, movie_id):
        return self.run("MATCH (m:Movie {movie_id:$id}) RETURN m.title AS title", id=movie_id)[0]["title"]
    def shared_entity_rows(self, seed_movie_id, sources):
        return self.run(SHARED_ENTITY_QUERY, seed=seed_movie_id, sources=list(sources))
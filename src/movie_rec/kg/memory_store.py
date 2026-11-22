from collections import defaultdict
class MemoryGraphStore:
    def __init__(self, graph):
        self.graph=graph
        self.postings=defaultdict(dict)
        for movie_id, edges in graph.edges.items():
            for (relation, entity), (source, confidence) in edges.items():
                self.postings[(relation, entity)][movie_id]=(source, confidence)
    def close(self):
        return None
    def has_movie(self, movie_id):
        return movie_id in self.graph.titles
    def movie_count(self):
        return len(self.graph.titles)
    def title(self, movie_id):
        return self.graph.titles[movie_id]
    def shared_entity_rows(self, seed_movie_id, sources):
        allowed=set(sources)
        degrees={}
        rows=[]
        for (relation, entity), (source, seed_confidence) in sorted(self.graph.edges[seed_movie_id].items()):
            if source not in allowed:
                continue
            members={movie_id:confidence for movie_id, (member_source, confidence) in self.postings[(relation, entity)].items() if member_source in allowed}
            for movie_id, confidence in members.items():
                if movie_id != seed_movie_id:
                    if movie_id not in degrees:
                        degrees[movie_id]=sum(1 for member_source, _ in self.graph.edges[movie_id].values() if member_source in allowed)
                    rows.append({
                        "candidate_id":movie_id,
                        "candidate_title":self.graph.titles[movie_id],
                        "relation":relation,
                        "entity":entity,
                        "entity_movie_count":len(members),
                        "seed_confidence":seed_confidence,
                        "candidate_confidence":confidence,
                        "candidate_degree":degrees[movie_id],
                    })
        return rows
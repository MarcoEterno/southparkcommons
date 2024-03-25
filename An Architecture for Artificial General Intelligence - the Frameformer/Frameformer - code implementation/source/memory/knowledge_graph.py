
class KnowledgeGraph:
    def __init__(self):
        self.graph = {}

    def add_relation(self, entity1, relation, entity2):
        if entity1 not in self.graph:
            self.graph[entity1] = []
        self.graph[entity1].append((relation, entity2))

    def get_relations(self, entity):
        return self.graph.get(entity, [])

    def remove_entity(self, entity):
        if entity in self.graph:
            del self.graph[entity]
        for relations in self.graph.values():
            relations[:] = [rel for rel in relationss if rel[1] != entity]

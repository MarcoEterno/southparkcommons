
import numpy as np

class VectorDatabase:
    def __init__(self):
        self.vectors = []
        self.ids = []

    def add_vector(self, vector_id, vector):
        self.ids.append(vector_id)
        self.vectors.append(vector)

    def get_vector(self, vector_id):
        if vector_id in self.ids:
            index = self.ids.index(vector_id)
            return self.vectors[index]
        return None

    def remove_vector(self, vector_id):
        if vector_id in self.ids:
            index = self.ids.index(vector_id)
            del self.ids[index]
            del self.vectors[index]

    def search_similar_vectors(self, query_vector, top_k=1):
        # Simple Euclidean distance for demonstration
        distances = [np.linalg.norm(np.array(vector) - np.array(query_vector)) for vector in self.vectors]
        sorted_indices = np.argsort(distances)[:top_k]
        return [self.ids[i] for i in sorted_indices]

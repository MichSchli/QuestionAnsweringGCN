import numpy as np


class CandidateGraph:

    vertices = None
    edges = None
    sentence_entities = None

    def __init__(self, sentence_entities, vertices, edges):
        self.sentence_entities = sentence_entities
        self.vertices = vertices
        self.edges = edges

    def get_candidate_vertices(self):
        true_candidates = np.isin(self.vertices, self.sentence_entities, invert=True)
        return self.vertices[true_candidates]
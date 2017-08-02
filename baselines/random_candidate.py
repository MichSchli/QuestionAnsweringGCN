import numpy as np


class RandomCandidate:

    candidate_neighborhood_generator = None

    def __init__(self, candidate_neighborhood_generator):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator

    def predict(self, filename):
        for candidate_graph in self.candidate_neighborhood_generator.parse_file(filename):
            yield np.random.choice(candidate_graph.get_candidate_vertices())

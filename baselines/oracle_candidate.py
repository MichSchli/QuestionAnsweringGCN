import numpy as np


class OracleCandidate:

    candidate_neighborhood_generator = None
    gold_generator = None

    def __init__(self, candidate_neighborhood_generator, gold_generator):
        self.candidate_neighborhood_generator = candidate_neighborhood_generator
        self.gold_generator = gold_generator

    def parse_file(self, filename):
        candidate_iterator = self.candidate_neighborhood_generator.parse_file(filename)
        gold_iterator = self.gold_generator.parse_file(filename, output="gold")

        for candidate_graph, gold_set in zip(candidate_iterator, gold_iterator):
            candidate_set = candidate_graph.vertices
            #print(candidate_set.shape)
            gold_in_candidates = np.isin(gold_set, candidate_set)
            yield gold_set[gold_in_candidates]


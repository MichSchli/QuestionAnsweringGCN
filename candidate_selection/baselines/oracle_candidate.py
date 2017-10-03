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

        counter = 1
        print("Running oracle...")
        for candidate_graph, gold_set in zip(candidate_iterator, gold_iterator):
            #print("aiololo")
            print(str(counter))
            counter += 1
            candidate_set = candidate_graph.get_vertices(type="entities")
            gold_in_candidates = np.isin(gold_set, candidate_set)
            print(candidate_set)
            print(gold_set)
            print(gold_in_candidates)
            yield gold_set[gold_in_candidates]

    def train(self, train_file):
        pass


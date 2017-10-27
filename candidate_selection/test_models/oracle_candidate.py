import numpy as np


class OracleCandidate:

    candidate_neighborhood_generator = None
    gold_generator = None

    is_tensorflow = False

    def set_neighborhood_generate(self, neighborhood_generator):
        self.candidate_neighborhood_generator = neighborhood_generator

    def update_setting(self, name, value):
        pass

    def initialize(self):
        pass

    def predict(self, test_file_iterator):
        epoch_iterator = test_file_iterator.iterate()
        epoch_iterator = self.candidate_neighborhood_generator.enrich(epoch_iterator)

        for element in epoch_iterator:
            candidate_set = element["neighborhood"].get_vertices(type="entities")
            gold_set = element["gold_entities"]
            gold_in_candidates = np.isin(gold_set, candidate_set)
            yield gold_set[gold_in_candidates]

    def train(self, train_file):
        pass


import numpy as np


class OracleCandidate:

    candidate_neighborhood_generator = None
    gold_generator = None

    is_tensorflow = False

    def __init__(self, facts):
        pass

    def set_neighborhood_generate(self, neighborhood_generator):
        self.candidate_neighborhood_generator = neighborhood_generator

    def set_indexers(self, indexers):
        pass

    def set_preprocessor(self, preprocessor):
        pass

    def update_setting(self, name, value):
        pass

    def initialize(self):
        pass

    def to_answer(self, example, index):
        if example["neighborhood"].has_name(index):
            return example["neighborhood"].get_name(index)
        else:
            return example["neighborhood"].from_index(index)

    def predict(self, element):
        candidate_set = element["neighborhood"].get_vertices(type="entities")
        candidate_set = [element["neighborhood"].from_index(i) for i in range(candidate_set.shape[0])]
        gold_set = element["gold_entities"]
        gold_in_candidates = np.isin(gold_set, candidate_set)
        return gold_set[gold_in_candidates]

    def train(self, train_file):
        pass


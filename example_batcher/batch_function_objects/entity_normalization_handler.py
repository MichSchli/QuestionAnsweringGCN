import numpy as np


class EntityNormalizationHandler:

    batch = None

    def __init__(self, batch):
        self.batch = batch


    def get_normalization_by_vertex_count(self, weight_positives):
        vertex_lists = [np.ones_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.batch.examples]

        for i,example in enumerate(self.batch.examples):
            vertex_lists[i] /= (example.count_entities() * self.batch.count_examples())

            if weight_positives and example.get_gold_indexes().shape[0] > 0:
                weight = example.count_entities() / example.get_gold_indexes().shape[0]
                for gold_index in example.get_gold_indexes():
                    if gold_index >=0:
                        vertex_lists[i][gold_index] *= weight

        return np.concatenate(vertex_lists)
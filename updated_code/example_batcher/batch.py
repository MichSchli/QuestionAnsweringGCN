import numpy as np


class Batch:

    examples = None

    def __init__(self):
        self.examples = []

    def count_examples(self):
        return len(self.examples)

    def has_examples(self):
        return len(self.examples) > 0

    def get_examples(self):
        return self.examples

    def count_all_entities(self):
        return sum(example.count_entities() for example in self.examples)

    def count_all_vertices(self):
        return sum(example.count_vertices() for example in self.examples)

    def get_combined_vertex_indexes(self):
        index_lists = [np.copy(example.get_entity_vertex_indexes()) for example in self.examples]

        for i,example in enumerate(self.examples):
            if i > 0:
                index_lists[i-1] += example.count_vertices()

        return np.concatenate(index_lists)

    def get_gold_vector(self):
        vertex_lists = [np.zeros_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.examples]

        for i,example in enumerate(self.examples):
            for gold_answer in example.gold_answers:
                vertex_lists[i][gold_answer.entity_indexes] = 1

        return np.concatenate(vertex_lists)

    def get_normalization_by_vertex_count(self):
        vertex_lists = [np.ones_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.examples]

        for i,example in enumerate(self.examples):
            vertex_lists[i] *= (example.count_entities() * self.count_examples())

        return np.concatenate(vertex_lists)
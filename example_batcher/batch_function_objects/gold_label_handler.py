import numpy as np

class GoldLabelHandler:

    batch = None

    def __init__(self, batch):
        self.batch = batch

    def get_gold_vector_use_only_entities(self):
        vertex_lists = [np.zeros_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.batch.examples]

        for i,example in enumerate(self.batch.examples):
            for gold_index in example.get_gold_indexes():
                if gold_index >=0:
                    entity_vertex_list_index = example.graph.map_general_vertex_to_entity_index(gold_index)
                    vertex_lists[i][entity_vertex_list_index] = 1

        return np.concatenate(vertex_lists)

    def get_gold_vector_use_all_vertices(self):
        vertex_lists = [np.zeros_like(example.graph.vertices, dtype=np.float32) for example in self.batch.examples]

        for i,example in enumerate(self.batch.examples):
            for gold_index in example.get_gold_indexes():
                if gold_index >=0:
                    vertex_lists[i][gold_index] = 1

        return np.concatenate(vertex_lists)

    def get_padded_gold_matrix(self):
        max_entity_count = max(example.count_entities() for example in self.batch.examples)
        matrix = np.zeros((len(self.batch.examples), max_entity_count), dtype=np.int32)
        for i,example in enumerate(self.batch.examples):
            for gold_index in example.get_gold_indexes():
                if gold_index >=0:
                    matrix[i][gold_index] = 1

        return matrix
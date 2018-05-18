import numpy as np


class VertexIndexCombiner:

    batch = None

    def __init__(self, batch):
        self.batch = batch

    def get_combined_mention_dummy_indices(self):
        index_lists = [np.copy([m.dummy_index for m in example.mentions]) for example in self.batch.examples]
        return self.combine_and_shift_index_lists(index_lists)

    def get_combined_word_vertex_indices(self):
        index_lists = [np.copy(example.question.dummy_indexes) for example in self.batch.examples]
        return self.combine_and_shift_index_lists(index_lists)

    def get_combined_entity_vertex_map_indexes(self):
        index_lists = [np.copy(example.get_entity_vertex_indexes()) for example in self.batch.examples]
        return self.combine_and_shift_index_lists(index_lists)

    def get_combined_sender_indices(self):
        index_lists = [np.copy(example.graph.edges[:,0]) for example in self.batch.examples]
        return self.combine_and_shift_index_lists(index_lists)

    def get_combined_receiver_indices(self):
        index_lists = [np.copy(example.graph.edges[:,2]) for example in self.batch.examples]
        return self.combine_and_shift_index_lists(index_lists)

    def get_combined_sentence_vertex_indices(self):
        index_lists = np.array([[example.graph.get_sentence_vertex_index()] for example in self.batch.examples], dtype=np.int32)
        return self.combine_and_shift_index_lists(index_lists)

    def combine_and_shift_index_lists(self, index_lists):
        accumulator = 0
        for i, example in enumerate(self.batch.examples):
            index_lists[i] += accumulator
            accumulator += example.count_vertices()
        return np.concatenate(index_lists)

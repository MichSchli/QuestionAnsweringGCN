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

    def get_combined_sender_indices(self):
        index_lists = [np.copy(example.graph.edges[:,0]) for example in self.examples]

        for i,example in enumerate(self.examples):
            if i > 0:
                index_lists[i-1] += example.count_vertices()

        return np.concatenate(index_lists)

    def get_combined_receiver_indices(self):
        index_lists = [np.copy(example.graph.edges[:,2]) for example in self.examples]

        for i,example in enumerate(self.examples):
            if i > 0:
                index_lists[i-1] += example.count_vertices()

        return np.concatenate(index_lists)

    def get_combined_edge_type_indices(self):
        index_lists = [np.copy(example.graph.edges[:,1]) for example in self.examples]

        return np.concatenate(index_lists)

    def get_gold_vector(self):
        vertex_lists = [np.zeros_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.examples]

        for i,example in enumerate(self.examples):
            for gold_answer in example.gold_answers:
                if gold_answer.entity_indexes >=0:
                    vertex_lists[i][gold_answer.entity_indexes] = 1

        return np.concatenate(vertex_lists)

    def get_normalization_by_vertex_count(self, weight_positives):
        vertex_lists = [np.ones_like(example.get_entity_vertex_indexes(), dtype=np.float32) for example in self.examples]

        for i,example in enumerate(self.examples):
            vertex_lists[i] /= (example.count_entities() * self.count_examples())

            if weight_positives:
                weight = example.count_entities() / example.get_gold_indexes().shape[0]
                for gold_index in example.get_gold_indexes():
                    if gold_index >=0:
                        vertex_lists[i][gold_index] *= weight

        return np.concatenate(vertex_lists)

    def get_padded_word_indices(self):
        max_word_count = max(example.count_words() for example in self.examples)
        sentence_matrix = np.zeros((len(self.examples), max_word_count), dtype=np.int32)

        for i,example in enumerate(self.examples):
            for j, word_index in enumerate(example.question.get_word_indexes()):
                sentence_matrix[i][j] = word_index

        return sentence_matrix

    def get_padded_pos_indices(self):
        max_word_count = max(example.count_words() for example in self.examples)
        sentence_matrix = np.zeros((len(self.examples), max_word_count), dtype=np.int32)

        for i,example in enumerate(self.examples):
            for j, word_index in enumerate(example.question.get_pos_indexes()):
                sentence_matrix[i][j] = word_index

        return sentence_matrix

    def get_word_padding_mask(self):
        max_word_count = max(example.count_words() for example in self.examples)
        sentence_matrix = np.zeros((len(self.examples), max_word_count), dtype=np.float32)

        for i,example in enumerate(self.examples):
            for j, word_index in enumerate(example.question.get_word_indexes()):
                sentence_matrix[i][j] = 1.0

        return sentence_matrix

    def get_padded_edge_part_type_matrix(self):
        pad_to = max(example.get_padded_edge_part_type_matrix().shape[1] for example in self.examples)
        index_lists = []

        for i,example in enumerate(self.examples):
            example_bags = example.get_padded_edge_part_type_matrix()
            padding_needed = pad_to - example_bags.shape[1]

            if padding_needed > 0:
                example_bags = np.pad(example_bags, ((0,0), (0,padding_needed)), mode='constant')

            index_lists.append(example_bags)

        return np.concatenate(index_lists)
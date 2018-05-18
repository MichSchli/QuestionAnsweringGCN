import numpy as np


class SentenceFeatureCombiner:

    batch = None

    def __init__(self, batch):
        self.batch = batch

    def get_padded_pos_indices(self):
        sentence_matrix = self.get_empty_sentence_matrix()
        features = [example.question.get_pos_indexes() for example in self.batch.examples]
        self.distribute_to_matrix(features, sentence_matrix)

        return sentence_matrix

    def get_padded_word_indices(self):
        sentence_matrix = self.get_empty_sentence_matrix()
        features = [example.question.get_word_indexes() for example in self.batch.examples]
        self.distribute_to_matrix(features, sentence_matrix)

        return sentence_matrix

    def get_word_padding_mask(self):
        sentence_matrix = self.get_empty_sentence_matrix()
        features = [np.ones_like(example.question.get_word_indexes()) for example in self.batch.examples]
        self.distribute_to_matrix(features, sentence_matrix)

        return sentence_matrix.astype(np.float32)

    def get_empty_sentence_matrix(self):
        max_word_count = max(example.count_words() for example in self.batch.examples)
        sentence_matrix = np.zeros((len(self.batch.examples), max_word_count), dtype=np.int32)
        return sentence_matrix

    def get_padded_mention_indicators(self):
        sentence_matrix = self.get_empty_sentence_matrix()

        for i,example in enumerate(self.batch.examples):
            for m in example.mentions:
                sentence_matrix[i][m.word_indexes] = 1.0

        return sentence_matrix.astype(np.float32)

    def distribute_to_matrix(self, features, sentence_matrix):
        for i, feature_list in enumerate(features):
            for j, feature in enumerate(feature_list):
                sentence_matrix[i][j] = feature
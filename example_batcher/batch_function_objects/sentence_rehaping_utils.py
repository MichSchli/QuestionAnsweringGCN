import numpy as np

class SentenceReshapingUtils:

    def __init__(self, batch):
        self.batch = batch

    def get_flat_word_indexes_arranged_in_padded_matrix(self):
        max_word_count = max(example.count_words() for example in self.batch.examples)
        matrix = np.zeros((self.batch.count_examples(), max_word_count), dtype=np.int32)

        accumulator = 1
        for i, example in enumerate(self.batch.examples):
            matrix[i][:example.count_words()] = np.arange(example.count_words()) + accumulator
            accumulator += example.count_words()

        return matrix

    def get_word_matrix_indexes_arranged_flat(self):
        max_word_count = max(example.count_words() for example in self.batch.examples)
        index_lists = [np.arange(len(example.question.dummy_indexes)) for example in self.batch.examples]

        accumulator = 0
        for i, example in enumerate(self.batch.examples):
            index_lists[i] += accumulator
            accumulator += max_word_count

        full = np.concatenate(index_lists)
        return full
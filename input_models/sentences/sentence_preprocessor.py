from candidate_selection.models.lazy_indexer import LazyIndexer
import numpy as np

from input_models.sentences.sentence_input_model import SentenceInputModel


class SentencePreprocessor():

    indexer = None

    def __init__(self):
        self.indexer = LazyIndexer()

    def preprocess(self, sentence_batch):
        words = [[self.indexer.index_single_element(word[1]) for word in sentence] for sentence in sentence_batch]
        # convert sentence batch to vocabulary
        batch_indices = -1 * np.ones((len(words), max([len(w) for w in words])), dtype=np.int32)
        sentence_lengths = np.zeros(len(words), dtype=np.int32)

        for i in range(len(words)):
            sentence_lengths[i] = len(words[i])
            for j in range(len(words[i])):
                batch_indices[i,j] = words[i][j]

        sentence_input_model = SentenceInputModel()
        sentence_input_model.word_index_matrix = batch_indices + 1
        sentence_input_model.sentence_lengths = sentence_lengths

        return sentence_input_model
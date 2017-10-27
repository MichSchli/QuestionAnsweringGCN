import numpy as np

from indexing.lazy_indexer import LazyIndexer
from input_models.abstract_preprocessor import AbstractPreprocessor
from input_models.sentences.sentence_input_model import SentenceInputModel


class SentencePreprocessor(AbstractPreprocessor):

    indexer = None
    sentence_string = "sentence"

    def __init__(self, indexer, sentence_string, output_name, next_preprocessor, clean_dictionary=True):
        AbstractPreprocessor.__init__(self, next_preprocessor)
        self.indexer = indexer
        self.sentence_string = sentence_string
        self.clean_dictionary = clean_dictionary
        self.output_name = output_name

    def process(self, batch_dictionary, mode='train'):
        if self.next_preprocessor is not None:
            self.next_preprocessor.process(batch_dictionary, mode=mode)

        sentence_batch = batch_dictionary[self.sentence_string]

        if self.clean_dictionary:
            del batch_dictionary[self.sentence_string]

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

        batch_dictionary[self.output_name] = sentence_input_model
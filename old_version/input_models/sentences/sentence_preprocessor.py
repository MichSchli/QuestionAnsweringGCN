import numpy as np

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
        centroid_map = batch_dictionary["sentence_entity_map"]

        if self.clean_dictionary:
            del batch_dictionary[self.sentence_string]

        words = [[self.indexer.index_single_element(word[1]) for word in sentence] for sentence in sentence_batch]

        # convert sentence batch to vocabulary
        max_sentence_len = max([len(w) for w in words])
        batch_indices = np.zeros((len(words), max_sentence_len), dtype=np.int32)
        entity_mask = np.zeros((len(words), max_sentence_len), dtype=np.int32)
        sentence_lengths = np.zeros(len(words), dtype=np.int32)
        flat_entity_indices_in_sentence = []

        for i in range(len(words)):
            sentence_lengths[i] = len(words[i])
            for j in range(len(words[i])):
                batch_indices[i,j] = words[i][j]

            for entity in centroid_map[i]:
                start = int(entity[0])
                end = int(entity[1])
                entity_mask[i, start:end+1] = 1

                flat_entity_indices_in_sentence.extend(range(i*max_sentence_len+start,i*max_sentence_len+end+1))


        sentence_input_model = SentenceInputModel()
        sentence_input_model.word_index_matrix = batch_indices
        sentence_input_model.sentence_lengths = sentence_lengths

        sentence_input_model.entity_mask = entity_mask
        sentence_input_model.flat_entity_indices_in_sentence = np.array(flat_entity_indices_in_sentence).astype(np.int32)

        batch_dictionary[self.output_name] = sentence_input_model
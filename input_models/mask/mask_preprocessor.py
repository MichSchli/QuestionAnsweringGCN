from input_models.abstract_preprocessor import AbstractPreprocessor
import numpy as np

class LookupMaskPreprocessor(AbstractPreprocessor):

    target_structure = None
    target_tensor = None

    def __init__(self, target_structure, target_tensor, input_string, output_string, next_preprocessor, clean_dictionary=True):
        AbstractPreprocessor.__init__(self, next_preprocessor)
        self.target_structure = target_structure
        self.target_tensor = target_tensor
        self.input_string = input_string
        self.output_string = output_string
        self.clean_dictionary = clean_dictionary

    def process(self, batch_dictionary, mode='train'):
        if self.next_preprocessor is not None:
            self.next_preprocessor.process(batch_dictionary, mode=mode)

        target = batch_dictionary[self.target_structure]

        mask = np.zeros_like(target.get(self.target_tensor), dtype=np.float32)
        for i, lookups in enumerate(batch_dictionary[self.input_string]):
            lookup_indexes = np.array([target.retrieve_index_in_batch(i, lookup) for lookup in lookups])
            actual_indices = lookup_indexes if i == 0 else lookup_indexes - np.max(target.get(self.target_tensor)[i-1])
            mask[i][actual_indices] = 1

        if self.clean_dictionary:
            del batch_dictionary[self.input_string]
        batch_dictionary[self.output_string] = mask
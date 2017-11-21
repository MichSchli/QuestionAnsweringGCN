from input_models.abstract_preprocessor import AbstractPreprocessor
import numpy as np

class LookupMaskPreprocessor(AbstractPreprocessor):

    target_structure = None
    target_tensor = None
    mode = None

    def __init__(self, target_structure, target_tensor, input_string, output_string, next_preprocessor, clean_dictionary=True, mode=None):
        AbstractPreprocessor.__init__(self, next_preprocessor)
        self.target_structure = target_structure
        self.target_tensor = target_tensor
        self.input_string = input_string
        self.output_string = output_string
        self.clean_dictionary = clean_dictionary
        self.mode = mode

    def process(self, batch_dictionary, mode='train'):
        if self.next_preprocessor is not None:
            self.next_preprocessor.process(batch_dictionary, mode=mode)

        if self.mode is not None and mode != self.mode:
            return

        target = batch_dictionary[self.target_structure]

        mask = np.zeros_like(target.get(self.target_tensor), dtype=np.float32)
        
        #print(self.input_string)
        #print(batch_dictionary[self.input_string])

        for i, lookups in enumerate(batch_dictionary[self.input_string]):
            mask[i][np.array(lookups)] = 1

        if self.clean_dictionary:
            del batch_dictionary[self.input_string]
        batch_dictionary[self.output_string] = mask

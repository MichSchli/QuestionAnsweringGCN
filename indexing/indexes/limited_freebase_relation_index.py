from indexing.abstract_index import AbstractIndex
import numpy as np


class LimitedFreebaseRelationIndex(AbstractIndex):

    additional_vector_count = 100
    cutoff = 20

    def __init__(self, dimension, keep_space_for_inverse):
        AbstractIndex.__init__(self, None, dimension)
        self.keep_space_for_inverse = keep_space_for_inverse
        self.index("<dummy_to_mention>")
        self.index("<dummy_to_word>")
        self.index("<word_to_word>")
        self.index("<sentence_to_word>")

        self.index_dependency_labels()

        self.load_file()

    def get_file_string(self):
        return "data/webquestions/edge_count.txt"

    def load_file(self):
        file_string = self.get_file_string()

        for line in open(file_string, encoding="utf8"):
            parts = line.strip().split(" ")

            count = int(parts[0].strip())
            label = parts[1].strip()

            if count > self.cutoff:
                self.index(label)

        self.vector_count = (self.additional_vector_count + self.element_counter) * (2 if self.keep_space_for_inverse else 1)
        self.inverse_edge_delimiter = self.additional_vector_count + self.element_counter

        self.freeze()
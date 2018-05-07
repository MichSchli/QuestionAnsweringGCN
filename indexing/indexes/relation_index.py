from indexing.abstract_index import AbstractIndex
import numpy as np


class RelationIndex(AbstractIndex):

    vector_count = 500
    inverse_edge_delimiter = 250

    def __init__(self, index_cache_name, dimension, keep_space_for_inverse=False):
        AbstractIndex.__init__(self, index_cache_name, dimension)
        self.index("<dummy_to_mention>")
        self.index("<dummy_to_word>")
        self.index("<word_to_word>")
        self.index("<sentence_to_word>")

        self.index_dependency_labels()
import numpy as np

from experiment_construction.index_construction.abstract_indexer import AbstractIndexer


class LazyIndexer(AbstractIndexer):

    global_map = None
    inverted_map = []
    counter = None
    vocabulary_shape = None
    vectors = None

    is_frozen = False

    def __init__(self, vocabulary_shape):
        self.global_map = {}
        self.counter = 0
        self.vocabulary_shape = np.array([vocabulary_shape[0] + 2, vocabulary_shape[1]])
        self.index_single_element("<unknown>")
        self.index_single_element("<dummy>")
        self.is_frozen = False

    def index_single_element(self, element):
        if element in self.global_map:
            return self.global_map[element]
        elif self.is_frozen:
            return 0
        else:
            self.global_map[element] = self.counter
            self.inverted_map.append(element)
            self.counter += 1
            return self.global_map[element]

    def invert(self, index):
        return self.inverted_map[index]

    def get_dimension(self):
        return self.vocabulary_shape[1]

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        if self.vectors is None:
            self.vectors = np.random.uniform(-1, 1, size=self.vocabulary_shape).astype(np.float32)
            self.vectors[0] = 0
        return self.vectors

    def index(self, elements):
        local_map = np.empty(elements.shape, dtype=np.int32)

        for i, element in enumerate(elements):
            if element in self.global_map:
                local_map[i] = self.global_map[element]
            elif self.is_frozen:
                local_map[i] = 0
            else:
                self.global_map[element] = self.counter
                self.inverted_map.append(element)
                local_map[i] = self.counter
                self.counter += 1

        return local_map
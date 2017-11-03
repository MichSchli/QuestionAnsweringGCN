import numpy as np


class LazyIndexer:

    global_map = None
    counter = None
    vocabulary_shape = None

    def __init__(self, vocabulary_shape):
        self.global_map = {}
        self.counter = 0
        self.vocabulary_shape = vocabulary_shape

    def index_single_element(self, element):
        if element not in self.global_map:
            self.global_map[element] = self.counter
            self.counter += 1

        return self.global_map[element]

    def get_dimension(self):
        return self.vocabulary_shape[1]

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        return np.random.normal(0, 0.01, size=self.vocabulary_shape).astype(np.float32)

    def index(self, elements):
        local_map = np.empty(elements.shape, dtype=np.int32)

        for i, element in enumerate(elements):
            if element in self.global_map:
                local_map[i] = self.global_map[element]
            else:
                self.global_map[element] = self.counter
                local_map[i] = self.counter
                self.counter += 1

        return local_map
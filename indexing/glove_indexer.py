import numpy as np

from indexing.lazy_indexer import LazyIndexer


class GloveIndexer:

    vectors = None

    def __init__(self, dimension):
        self.dimension = dimension

        self.load_file()

    def get_dimension(self):
        return self.dimension

    def index(self, elements):
        local_map = np.empty(elements.shape, dtype=np.int32)

        for i, element in enumerate(elements):
            local_map[i] = self.index_single_element(element)

        return local_map

    def index_single_element(self, element):
        if element not in self.indexer.global_map:
            return 0
        else:
            return self.indexer.global_map[element]

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        return self.vectors

    def load_file(self):
        file_string = "data/glove.6B/glove.6B." + str(self.dimension) + "d.txt"
        counter = 0

        num_lines = sum(1 for _ in open(file_string, encoding="utf8"))
        self.vectors = np.empty((num_lines+1, self.dimension), dtype=np.float32)
        self.vectors[0] = 0

        self.indexer = LazyIndexer((num_lines, self.dimension))

        for line in open(file_string, encoding="utf8"):
            counter += 1
            parts = line.strip().split(" ")
            self.indexer.index_single_element(parts[0])
            self.vectors[counter] = parts[1:]
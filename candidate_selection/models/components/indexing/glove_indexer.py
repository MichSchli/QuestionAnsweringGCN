from candidate_selection.models.lazy_indexer import LazyIndexer
import numpy as np


class GloveIndexer:

    vectors = None

    def __init__(self, dimension):
        self.indexer = LazyIndexer()
        self.dimension = dimension
        self.indexer.index_single_element("<unknown>")

        self.load_file()

    def index_single_element(self, element):
        if element not in self.indexer.global_map:
            return 0
        else:
            return self.indexer.global_map[element]

    def get_all_vectors(self):
        return self.vectors

    def load_file(self):
        file_string = "data/glove.6B/glove.6B." + str(self.dimension) + "d.txt"
        counter = 0

        num_lines = sum(1 for _ in open(file_string, encoding="utf8"))
        self.vectors = np.empty((num_lines+1, self.dimension), dtype=np.float32)
        self.vectors[0] = np.random.uniform(-1, 1, self.dimension)

        for line in open(file_string, encoding="utf8"):
            counter += 1
            parts = line.strip().split(" ")
            self.indexer.index_single_element(parts[0])
            self.vectors[counter] = parts[1:]
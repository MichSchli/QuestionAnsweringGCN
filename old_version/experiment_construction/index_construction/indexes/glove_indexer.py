import numpy as np

from experiment_construction.index_construction.abstract_indexer import AbstractIndexer
from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer


class GloveIndexer(AbstractIndexer):

    vectors = None

    def __init__(self, dimension):
        self.dimension = dimension
        self.vocab_size = self.get_vocab_size()
        inner = LazyIndexer((self.vocab_size, self.dimension))
        AbstractIndexer.__init__(self, inner)

        self.load_file()

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        return self.vectors

    def get_file_string(self):
        return "data/glove.6B/glove.6B." + str(self.dimension) + "d.txt"

    def get_vocab_size(self):
        return sum(1 for _ in open(self.get_file_string(), encoding="utf8"))

    def load_file(self):
        file_string = self.get_file_string()
        counter = 0

        self.vectors = np.empty((self.vocab_size+1, self.dimension), dtype=np.float32)
        self.vectors[0] = 0

        for line in open(file_string, encoding="utf8"):
            counter += 1
            parts = line.strip().split(" ")
            self.inner.index_single_element(parts[0])
            self.vectors[counter] = parts[1:]

        self.inner.freeze()
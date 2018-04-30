from indexing.abstract_index import AbstractIndex
import numpy as np


class DepIndex(AbstractIndex):

    vectors = None

    def __init__(self):
        self.dimension = 300
        AbstractIndex.__init__(self, None, self.dimension)

        self.vocab_size = self.get_vocab_size()
        self.load_file()

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        return self.vectors

    def get_file_string(self):
        return "data/embeddings/deps.words"

    def index(self, element):
        element = element.lower()
        return AbstractIndex.index(self, element)

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
            self.index(parts[0])
            self.vectors[counter] = [float(p) for p in parts[1:]]

        self.freeze()
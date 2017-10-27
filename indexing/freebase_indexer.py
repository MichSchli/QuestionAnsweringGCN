import numpy as np

from indexing.lazy_indexer import LazyIndexer


class FreebaseIndexer:

    vectors = None

    def __init__(self):
        self.dimension = 1000

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

    def get_all_vectors(self):
        return self.vectors

    def load_file(self):
        file_string = "/home/michael/Projects/QuestionAnswering/GCNQA/data/embeddings/freebase-vectors-skipgram1000.bin"
        counter = 0

        num_lines = sum(1 for _ in open(file_string, "rb"))
        #self.vectors = np.empty((num_lines+1, self.dimension), dtype=np.float32)
        #self.vectors[0] = np.random.uniform(-1, 0.01, self.dimension)

        #self.indexer = LazyIndexer((num_lines+1, self.dimension))
        #self.indexer.index_single_element("<unknown>")

        print("init")

        for line in open(file_string, "rb"):
            counter += 1
            parts = line.strip().split(" ")
            print(parts[1])
            exit()
            self.indexer.index_single_element(parts[0])
            self.vectors[counter] = parts[1:]


FreebaseIndexer()
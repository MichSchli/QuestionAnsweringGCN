from diskcache import Cache
import numpy as np

class AbstractIndex:

    forward_map = None
    backward_map = None
    is_frozen = None
    element_counter = None
    vectors = None

    def __init__(self, index_cache_name, dimension):
        if index_cache_name is not None:
            index_cache_path = "indexing/cache/"+index_cache_name
            self.forward_map = Cache(index_cache_path + "/forward")
            self.backward_map = Cache(index_cache_path + "/backward")
            self.element_counter = len(self.forward_map)
        else:
            self.forward_map = {}
            self.backward_map = {}
            self.element_counter = 0

        self.dimension = dimension

        self.is_frozen = False

        self.index("<unknown>")

    def index(self, element):
        if element not in self.forward_map:
            if self.is_frozen:
                return self.forward_map["<unknown>"]
            
            self.forward_map[element] = self.element_counter
            self.backward_map[self.element_counter] = element
            self.element_counter += 1

        return self.forward_map[element]

    def get_all_vectors(self):
        if self.vectors is None:
            self.vectors = np.random.uniform(-1, 1, size=(self.vector_count, self.dimension)).astype(np.float32)
            self.vectors[0] = 0
        return self.vectors

    def freeze(self):
        self.is_frozen = True

    def from_index(self, index):
        return self.backward_map[index]

    def index_dependency_labels(self):
        filename = "data/dependency_labels.txt"

        for line in open(filename, "r"):
            self.index("<dep:"+line.strip()+">")
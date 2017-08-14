import numpy as np


class LazyIndexer:

    global_map = None
    counter = 0

    def __init__(self):
        self.global_map = {}
        self.counter = 0

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
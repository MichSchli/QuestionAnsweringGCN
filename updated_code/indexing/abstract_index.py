from diskcache import Cache

class AbstractIndex:

    forward_map = None
    backward_map = None
    is_frozen = None
    element_counter = None

    def __init__(self, index_cache_name):
        index_cache_path = "indexing/cache/"+index_cache_name
        self.forward_map = Cache(index_cache_path + "/forward")
        self.backward_map = Cache(index_cache_path + "/backward")
        self.element_counter = len(self.forward_map)
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

    def freeze(self):
        self.is_frozen = True

    def from_index(self, index):
        return self.backward_map[index]
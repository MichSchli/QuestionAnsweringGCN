from model.hypergraph_model import HypergraphModel


class CandidateGeneratorCache:

    keys = None
    memory = None
    row_pointer = None
    inner = None
    disk_cache_file = None

    def __init__(self, inner, disk_cache=None):
        self.keys = {}
        self.row_pointer = -1
        self.inner = inner

        if disk_cache is None:
            self.memory = []
        else:
            self.disk_cache_file=disk_cache
            self.index_cache()

    def index_cache(self):
        for line in open(self.disk_cache_file, "r"):
            if line.strip():
                parts = line.strip().split("\t")
                self.row_pointer += 1
                self.keys[parts[0]] = self.row_pointer

    def enrich(self, instances):
        for instance in instances:
            key = "cachekey_"+":".join(instance["mentioned_entities"])

            if key not in self.keys:
                neighborhood_hypergraph = self.inner.generate_neighborhood(instance)
                self.add_to_memory(key, neighborhood_hypergraph)
                instance["neighborhood"] = neighborhood_hypergraph
            else:
                instance["neighborhood"] = self.retrieve_from_memory(key)

            yield instance

    def add_to_memory(self, key, hypergraph):
        self.row_pointer += 1

        if self.memory is not None:
            self.memory.append(hypergraph)
        else:
            file = open(self.disk_cache_file, "a")
            string_representation = key + "\t" + hypergraph.to_string_storage()
            file.write(string_representation + "\n")

        self.keys[key] = self.row_pointer

    def retrieve_from_memory(self, key):
        location = self.keys[key]
        if self.memory is not None:
            return self.memory[location]
        else:
            line = self.retrieve_line_from_disk_memory(location)
            hypergraph = HypergraphModel()
            hypergraph.load_from_string_storage(line[len(key)+1:])
            return hypergraph

    def retrieve_line_from_disk_memory(self, line_number):
        if line_number < 0: return ''
        for current_line_number, line in enumerate(open(self.disk_cache_file, 'r')):
            if line.strip() and current_line_number == line_number:
                return line.strip()
        return ''

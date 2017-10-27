from model.hypergraph_model import HypergraphModel
from diskcache import Cache

class CandidateGeneratorCache:

    inner = None
    disk_cache_file = None

    def __init__(self, inner, disk_cache=None):
        self.keys = {}
        self.row_pointer = -1
        self.inner = inner

        if disk_cache is None:
            self.cache = {}
        else:
            self.disk_cache_file=disk_cache
            self.cache = Cache(self.disk_cache_file)

    def enrich(self, instances):
        for instance in instances:
            key = "cachekey_"+":".join(instance["mentioned_entities"])
            print(key)

            if key not in self.cache:
                print("computed")
                neighborhood_hypergraph = self.inner.generate_neighborhood(instance)
                self.cache[key] = neighborhood_hypergraph
                instance["neighborhood"] = neighborhood_hypergraph
            else:
                print("retrieved")
                instance["neighborhood"] = self.cache[key]

            yield instance
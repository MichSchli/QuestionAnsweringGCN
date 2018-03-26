from diskcache import Cache


class GraphCache:
    inner = None
    disk_cache_file = None

    def __init__(self, inner, disk_cache):
        self.keys = {}
        self.row_pointer = -1
        self.inner = inner

        if disk_cache is None:
            self.cache = {}
        else:
            self.disk_cache_file = disk_cache
            self.cache = Cache(self.disk_cache_file, size_limit=2 ** 42)

    def get_neighborhood_graph(self, mention_entities):
        key = "cachekey_" + ":".join(mention_entities)

        if key not in self.cache:
            print("retrieve")
            neighborhood_graph = self.inner.get_neighborhood_graph(mention_entities)
            self.cache[key] = neighborhood_graph

            return neighborhood_graph
        else:
            return self.cache[key]
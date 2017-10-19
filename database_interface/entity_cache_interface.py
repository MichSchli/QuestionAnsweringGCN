class EntityCacheInterface:

    inner = None
    store = {}

    def __init__(self, inner):
        self.inner = inner
        self.store = {}

    def get_neighborhood_hypergraph(self, vertices, hops=1, extra_literals=False):
        hypergraphs = []

        for vertex in vertices:
            if vertex not in self.store.keys():
                cache_miss_hypergraph = self.inner.get_neighborhood_hypergraph([vertex], hops=hops,
                                                                               extra_literals=extra_literals)
                self.store[vertex] = cache_miss_hypergraph
                hypergraphs.append(cache_miss_hypergraph)
            else:
                hypergraphs.append(self.store[vertex])

        for i in range(1,len(hypergraphs)):
            hypergraphs[0].join_other_hypergraph(hypergraphs[i])

        return hypergraphs[0]




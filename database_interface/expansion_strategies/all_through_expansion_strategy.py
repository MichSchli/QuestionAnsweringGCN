from database_interface.search_filters.prefix_filter import PrefixFilter


class AllThroughExpansionStrategy:

    filter = None

    def __init__(self):
        self.filter = PrefixFilter("http://rdf.freebase.com/ns/")

    def get_frontier(self, vertices):
        pass_vertices = vertices[self.filter.accepts(vertices)]
        return pass_vertices
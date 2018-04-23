from database_interface.search_filters.prefix_filter import PrefixFilter


class AllThroughExpansionStrategy:

    filter = None

    def __init__(self):
        pass

    def get_frontier(self, vertices):
        return vertices
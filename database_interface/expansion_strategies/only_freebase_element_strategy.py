from database_interface.search_filters.prefix_filter import PrefixFilter
import numpy as np


class OnlyFreebaseExpansionStrategy:

    filter = None

    def __init__(self):
        self.filter = PrefixFilter("http://rdf.freebase.com/ns/")

    def get_frontier(self, vertices):
        filter_acceptance_indices = self.filter.accepts(vertices)
        if filter_acceptance_indices.shape[0] == 0:
            return np.empty((0))

        pass_vertices = vertices[self.filter.accepts(vertices)]
        return pass_vertices
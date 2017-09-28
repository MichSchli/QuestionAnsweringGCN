from database_interface.search_filters.infix_filter import InfixFilter
from database_interface.search_filters.prefix_filter import PrefixFilter
import numpy as np


class OnlyFreebaseExpansionStrategy:

    prefix_filter = None
    infix_filter = None

    def __init__(self):
        self.prefix_filter = PrefixFilter("http://rdf.freebase.com/ns/")
        self.infix_filter = InfixFilter(".", 28)

    def get_frontier(self, vertices):
        prefix_filter_acceptance_indices = self.prefix_filter.accepts(vertices)
        infix_filter_acceptance_indices = self.infix_filter.accepts(vertices)

        all_acceptance_indices = np.logical_and(prefix_filter_acceptance_indices, infix_filter_acceptance_indices)
        if all_acceptance_indices.shape[0] == 0:
            return np.empty((0))

        pass_vertices = vertices[all_acceptance_indices]
        return pass_vertices

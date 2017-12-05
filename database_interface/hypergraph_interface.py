from model.hypergraph_model import HypergraphModel
from time import sleep

class HypergraphInterface:

    data_interface = None
    expansion_algorithm = None
    prefix = None

    def __init__(self, data_interface, expansion_algorithm, prefix=""):
        self.data_interface = data_interface
        self.expansion_algorithm = expansion_algorithm
        self.prefix = prefix

    """
    Retrieve the n-neighborhood of a hypergraph, potentially including surrounding literals as well.
    """
    def get_neighborhood_hypergraph(self, vertices, hops=1, extra_literals=False):
        hypergraph = HypergraphModel()
        hypergraph.add_vertices(vertices, type="entities")
        hypergraph.populate_discovered("entities")
        hypergraph.set_centroids(vertices)

        for i in range(hops):
            self.expand_hypergraph_to_one_neighborhood(hypergraph, use_event_edges=True, literals_only=False)

        if extra_literals:
            self.expand_hypergraph_to_one_neighborhood(hypergraph, use_event_edges=False, literals_only=True)

        return hypergraph

    """
    Expand a hypergraph to include the one-neighborhood of every previously unexpanded vertex.
    """
    def expand_hypergraph_to_one_neighborhood(self, hypergraph, use_event_edges=True, literals_only=False):
        candidates_for_expansion = hypergraph.get_expandable_vertices("entities", pop=True)
        filtered_frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        if filtered_frontier.shape[0] == 0:
            return

        self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "entities", literals_only=literals_only)

        if use_event_edges:
            self.expand_hypergraph_through_hyperedges(filtered_frontier, hypergraph, literals_only=literals_only)

        hypergraph.mark_expanded(candidates_for_expansion, "entities")
        hypergraph.populate_discovered("entities")

    """
    Expand a hypergraph through a precomputed frontier to any adjacent vertices lying along hyperedges.
    """
    def expand_hypergraph_through_hyperedges(self, filtered_frontier, hypergraph, literals_only=False):
        self.expand_hypergraph_to_event_vertices(filtered_frontier, hypergraph)
        unexpanded_event_vertices = hypergraph.get_expandable_vertices("events", pop=True)
        if unexpanded_event_vertices.shape[0] > 0:
            self.expand_hypergraph_from_data_interface(unexpanded_event_vertices,
                                                       hypergraph,
                                                       "events",
                                                       "entities",
                                                       literals_only=literals_only)
            hypergraph.mark_expanded(unexpanded_event_vertices, "events")

    """
    Expand a hypergraph through a precomputed frontier to any adjacent event vertices
    """
    def expand_hypergraph_to_event_vertices(self, filtered_frontier, hypergraph):
        self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "events")
        hypergraph.populate_discovered("events")

    """
    Expand a hypergraph in a specified manner using the hypergraph interface below
    """
    def expand_hypergraph_from_data_interface(self,
                                              frontier,
                                              hypergraph,
                                              sources,
                                              targets,
                                              literals_only=False):
        edge_query_result = self.data_interface.get_adjacent_edges(frontier, target=targets, literals_only=literals_only)
        hypergraph.expand(edge_query_result.get_forward_edges(),
                          edge_query_result.get_backward_edges(),
                          sources=sources,
                          targets=targets)
        hypergraph.add_discovered_vertices(edge_query_result.get_forward_edges(),
                                           edge_query_result.get_backward_edges(),
                                           type=targets)
        hypergraph.add_names(edge_query_result.get_name_map())

from model.hypergraph import Hypergraph
import numpy as np

from model.hypergraph_model import HypergraphModel


class HypergraphInterface:

    data_interface = None
    expansion_algorithm = None
    vertex_property_retriever = None

    def __init__(self, data_interface, expansion_algorithm, vertex_property_retriever):
        self.data_interface = data_interface
        self.expansion_algorithm = expansion_algorithm
        self.vertex_property_retriever = vertex_property_retriever

    def get_neighborhood_hypergraph(self, vertices, hops=1):
        hypergraph = HypergraphModel()
        hypergraph.add_vertices(vertices, type="entities")

        for i in range(hops):
            self.expand_hypergraph_to_one_neighborhood(hypergraph)

    def get_neighborhood_hypergraph_old(self, vertices, hops=1):
        hypergraph = Hypergraph()
        hypergraph.add_vertices(np.array([[v,"entity"] for v in vertices]))
        hypergraph.centroids = vertices

        properties = self.vertex_property_retriever.retrieve_properties(vertices, np.array(["entity" for _ in vertices]))
        hypergraph.set_vertex_properties(properties)

        for i in range(hops):
            self.expand_hypergraph_to_one_neighborhood(hypergraph)

        return hypergraph

    def expand_hypergraph_to_one_neighborhood(self, hypergraph):
        candidates_for_expansion = hypergraph.get_expandable_vertices("entities")
        frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        self.expand_hypergraph_from_data_interface(candidates_for_expansion, frontier, hypergraph, "entities", "events")
        self.expand_hypergraph_from_data_interface(candidates_for_expansion, frontier, hypergraph, "entities", "entities")
        hypergraph.clear_expandable_vertices(frontier, type="entities")

        unexpanded_event_vertices = hypergraph.get_expandable_vertices("events")
        if unexpanded_event_vertices.shape[0] > 0:
            self.expand_hypergraph_from_data_interface(unexpanded_event_vertices,
                                                       unexpanded_event_vertices,
                                                       hypergraph,
                                                       "events",
                                                       "entities")
            hypergraph.clear_expandable_vertices(unexpanded_event_vertices, type="events")

        print(hypergraph.get_seen_vertices())
        print(hypergraph.get_expandable_vertices())
        exit()

    def expand_hypergraph_from_data_interface(self,
                                              candidates_for_expansion,
                                              frontier,
                                              hypergraph,
                                              sources,
                                              targets):
        edge_query_result = self.data_interface.get_adjacent_edges(frontier, target=targets)
        print(edge_query_result)
        hypergraph.expand(candidates_for_expansion,
                          edge_query_result.get_forward_edges(),
                          edge_query_result.get_backward_edges(),
                          sources=sources,
                          targets=targets)

    def expand_hypergraph_to_one_neighborhood_old(self, hypergraph):
        candidates_for_expansion = hypergraph.pop_all_unexpanded_vertices()
        frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        edge_query_result = self.data_interface.get_adjacent_edges(frontier)

        hypergraph.expand_forward(edge_query_result.get_forward_edges(), expanding_from=candidates_for_expansion)
        hypergraph.expand_backward(edge_query_result.get_backward_edges(), expanding_from=candidates_for_expansion)
        hypergraph.mark_expanded(frontier)

        hypergraph.add_vertices(np.array(list(edge_query_result.vertices.items())))

        vertices_lacking_properties, types = hypergraph.get_all_unexpanded_vertices(include_types=True)
        properties = self.vertex_property_retriever.retrieve_properties(vertices_lacking_properties, types)
        hypergraph.set_vertex_properties(properties)
        
        unexpanded_hyperedges = hypergraph.pop_unexpanded_hyperedges()
        if unexpanded_hyperedges.shape[0] > 0:
            additional_edges = self.data_interface.get_adjacent_edges(unexpanded_hyperedges)

            hypergraph.expand_forward(additional_edges.get_forward_edges(), expanding_from=candidates_for_expansion)
            hypergraph.expand_backward(additional_edges.get_backward_edges(), expanding_from=candidates_for_expansion)
            hypergraph.mark_expanded(unexpanded_hyperedges)

            hypergraph.add_vertices(np.array(list(additional_edges.vertices.items())))

            vertices_lacking_properties, types = hypergraph.get_most_recently_added_vertices(include_types=True)
            properties = self.vertex_property_retriever.retrieve_properties(vertices_lacking_properties, types)
            hypergraph.set_vertex_properties(properties)

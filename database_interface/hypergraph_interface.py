from model.hypergraph import Hypergraph
import numpy as np


class HypergraphInterface:

    data_interface = None
    expansion_algorithm = None
    vertex_property_retriever = None

    def __init__(self, data_interface, expansion_algorithm, vertex_property_retriever):
        self.data_interface = data_interface
        self.expansion_algorithm = expansion_algorithm
        self.vertex_property_retriever = vertex_property_retriever

    def get_neighborhood_hypergraph(self, vertices, hops=1):
        hypergraph = Hypergraph()
        hypergraph.add_vertices(np.array([[v,"entity"] for v in vertices]))

        properties = self.vertex_property_retriever.retrieve_properties(vertices)
        hypergraph.set_vertex_properties(properties)

        for i in range(hops):
            self.expand_hypergraph_to_one_neighborhood(hypergraph)

        return hypergraph

    def expand_hypergraph_to_one_neighborhood(self, hypergraph):
        candidates_for_expansion = hypergraph.pop_all_unexpanded_vertices()
        frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        edge_query_result = self.data_interface.get_adjacent_edges(frontier)
        hypergraph.add_edges(edge_query_result.get_edges())
        hypergraph.add_vertices(np.array(list(edge_query_result.vertices.items())))

        vertices_lacking_properties = hypergraph.get_all_unexpanded_vertices()
        properties = self.vertex_property_retriever.retrieve_properties(vertices_lacking_properties)
        hypergraph.set_vertex_properties(properties)
        
        unexpanded_hyperedges = hypergraph.pop_unexpanded_hyperedges()
        if unexpanded_hyperedges.shape[0] > 0:
            additional_edges = self.data_interface.get_adjacent_edges(unexpanded_hyperedges)
            hypergraph.add_edges(additional_edges.get_edges())
            hypergraph.add_vertices(np.array(list(additional_edges.vertices.items())))

            vertices_lacking_properties = hypergraph.get_most_recently_added_vertices()
            properties = self.vertex_property_retriever.retrieve_properties(vertices_lacking_properties)
            hypergraph.set_vertex_properties(properties)

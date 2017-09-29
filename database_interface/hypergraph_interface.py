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

    def get_neighborhood_hypergraph(self, vertices, hops=1, extra_literals=False):
        print(vertices)
        hypergraph = HypergraphModel()
        hypergraph.add_vertices(vertices, type="entities")
        hypergraph.populate_discovered("entities")

        #print(hypergraph.get_expandable_vertices("entities", pop=False))

        for i in range(hops):
            self.expand_hypergraph_to_one_neighborhood(hypergraph)

        if extra_literals:
            self.expand_hypergraph_to_adjacent_literals(hypergraph)

        return hypergraph

    def expand_hypergraph_to_one_neighborhood(self, hypergraph):
        # In the long run, we may wish to select these through some smarter strategy:
        candidates_for_expansion = hypergraph.get_expandable_vertices("entities", pop=True)

        # Applies a filter to remove e.g. non-freebase vertices:
        filtered_frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        if not filtered_frontier.shape[0] > 0:
            return

        #print(filtered_frontier)
        self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "events")
        self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "entities")

        hypergraph.populate_discovered("events")
        unexpanded_event_vertices = hypergraph.get_expandable_vertices("events", pop=True)

        if unexpanded_event_vertices.shape[0] > 0:
            #print("PRINTING EVENT VERTICES")
            #for v in unexpanded_event_vertices:
            #    print(v)
            self.expand_hypergraph_from_data_interface(unexpanded_event_vertices,
                                                       hypergraph,
                                                       "events",
                                                       "entities")
            hypergraph.mark_expanded(unexpanded_event_vertices, "events")

        hypergraph.mark_expanded(candidates_for_expansion, "entities")
        hypergraph.populate_discovered("entities")

    def expand_hypergraph_to_adjacent_literals(self, hypergraph, use_event_edges=False):
        # In the long run, we may wish to select these through some smarter strategy:
        candidates_for_expansion = hypergraph.get_expandable_vertices("entities", pop=True)

        # Applies a filter to remove e.g. non-freebase vertices:
        filtered_frontier = self.expansion_algorithm.get_frontier(candidates_for_expansion)

        #print("PRINTING ENTITY VERTICES")
        #for v in filtered_frontier:
        #    print(v)

        if not filtered_frontier.shape[0] > 0:
            return

        self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "entities", literals_only=True)

        if use_event_edges:
            self.expand_hypergraph_from_data_interface(filtered_frontier, hypergraph, "entities", "events")

            hypergraph.populate_discovered("events")
            unexpanded_event_vertices = hypergraph.get_expandable_vertices("events", pop=True)

            if unexpanded_event_vertices.shape[0] > 0:
                self.expand_hypergraph_from_data_interface(unexpanded_event_vertices,
                                                           hypergraph,
                                                           "events",
                                                           "entities",
                                                           literals_only=True)
                hypergraph.mark_expanded(unexpanded_event_vertices, "events")

        hypergraph.mark_expanded(candidates_for_expansion, "entities")
        hypergraph.populate_discovered("entities")

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
        #print(edge_query_result.get_forward_edges().shape)
        #print(edge_query_result.get_backward_edges().shape)
        #for edge in edge_query_result.get_forward_edges():
        #    print(edge)
        
        #if edge_query_result.get_forward_edges().shape[0] == 50000:
        #    print("ending")
        #    exit()
        hypergraph.add_discovered_vertices(edge_query_result.get_forward_edges(),
                                           edge_query_result.get_backward_edges(),
                                           type=targets)

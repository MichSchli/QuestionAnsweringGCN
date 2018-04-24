from example_reader.graph_reader.graph import Graph
import numpy as np

class GraphConverter:

    hypergraph_interface = None

    def __init__(self, hypergraph_interface):
        self.hypergraph_interface = hypergraph_interface

    def get_neighborhood_graph(self, entities):
        hypergraph = self.hypergraph_interface.get_neighborhood_graph(entities)

        graph = Graph()
        graph.vertices = np.concatenate((hypergraph.entity_vertices, hypergraph.event_vertices))
        graph.entity_vertex_indexes = np.arange(hypergraph.entity_vertices.shape[0], dtype=np.int32)

        #graph.nearby_centroid_map = []

        #for vertex in hypergraph.entity_vertices:
        #    graph.nearby_centroid_map.append(hypergraph.get_nearby_centroids(vertex))
        #for vertex in hypergraph.event_vertices:
        #    graph.nearby_centroid_map.append(hypergraph.get_nearby_centroids(vertex))

        graph.edges = np.concatenate((hypergraph.entity_to_entity_edges,
                                      hypergraph.event_to_entity_edges,
                                     hypergraph.entity_to_event_edges))

        vertex_name_map = {hypergraph.to_index(k):v for k,v in hypergraph.name_map.feature_map.items()}
        graph.set_index_to_name_map(vertex_name_map)

        entity_vertex_types = np.array([[1,0,0,0,0,0] for _ in range(hypergraph.entity_vertices.shape[0])], dtype=np.float32)
        event_vertex_types = np.array([[0,1,0,0,0,0] for _ in range(hypergraph.event_vertices.shape[0])], dtype=np.float32)

        if entity_vertex_types.shape[0] == 0:
            entity_vertex_types = np.empty((0,6), dtype=np.float32)

        if event_vertex_types.shape[0] == 0:
            event_vertex_types = np.empty((0,6), dtype=np.float32)

        graph.vertex_types = np.concatenate((entity_vertex_types, event_vertex_types))
        graph.nearby_centroid_map = [hypergraph.nearby_centroid_map[entity] for entity in graph.vertices]

        return graph
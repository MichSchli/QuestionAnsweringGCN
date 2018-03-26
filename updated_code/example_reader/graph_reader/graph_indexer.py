import numpy as np


class GraphIndexer:

    vertex_indexer = None
    relation_indexer = None

    def __init__(self, inner, vertex_indexer, relation_indexer):
        self.inner = inner
        self.vertex_indexer = vertex_indexer
        self.relation_indexer = relation_indexer

    def get_neighborhood_graph(self, mention_entities):
        graph = self.inner.get_neighborhood_graph(mention_entities)

        local_vertex_indexes = {}
        for i,vertex in enumerate(graph.vertices):
            local_vertex_indexes[vertex] = i
            graph.vertices[i] = self.vertex_indexer.index(vertex)

        for j,edge in enumerate(graph.edges):
            graph.edges[j][0] = local_vertex_indexes[graph.edges[j][0]]
            graph.edges[j][1] = self.relation_indexer.index(graph.edges[j][1])
            graph.edges[j][2] = local_vertex_indexes[graph.edges[j][2]]

        graph.vertices = np.array(graph.vertices, dtype=np.int32)
        graph.edges = np.array(graph.edges, dtype=np.int32)
        graph.vertex_label_to_index_map = local_vertex_indexes

        graph.nearby_centroid_map = [[local_vertex_indexes[v] for v in vertex] for vertex in graph.nearby_centroid_map]

        return graph
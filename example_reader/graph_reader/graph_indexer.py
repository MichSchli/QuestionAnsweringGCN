import numpy as np


class GraphIndexer:

    vertex_indexer = None
    relation_indexer = None
    relation_part_indexer = None

    def __init__(self, inner, vertex_indexer, relation_indexer, relation_part_indexer):
        self.inner = inner
        self.vertex_indexer = vertex_indexer
        self.relation_indexer = relation_indexer
        self.relation_part_indexer = relation_part_indexer

    def get_neighborhood_graph(self, mention_entities):

        graph = self.inner.get_neighborhood_graph(mention_entities)

        local_vertex_indexes = {}
        for i,vertex in enumerate(graph.vertices):
            local_vertex_indexes[vertex] = i

        relation_bags = [None] * graph.edges.shape[0]
        largest_bag_size = 0
        for i in range(graph.edges.shape[0]):
            relation = graph.edges[i][1]
            true_relation_name = relation.split("/")[-1]
            bag = np.unique(true_relation_name.replace(".", "_").split("_"))

            if len(bag) > largest_bag_size:
                largest_bag_size = len(bag)

            relation_bags[i] = [self.relation_part_indexer.index(b) for b in bag]

        graph.padded_edge_bow_matrix = np.zeros((len(relation_bags), largest_bag_size), dtype=np.int32)
        for i in range(graph.edges.shape[0]):
            for j, index in enumerate(relation_bags[i]):
                graph.padded_edge_bow_matrix[i][j] = index

        for j,edge in enumerate(graph.edges):
            graph.edges[j][0] = local_vertex_indexes[graph.edges[j][0]]
            graph.edges[j][1] = self.relation_indexer.index(graph.edges[j][1])
            graph.edges[j][2] = local_vertex_indexes[graph.edges[j][2]]

        graph.edges = np.array(graph.edges, dtype=np.int32)
        graph.set_label_to_index_map(local_vertex_indexes)

        #graph.map_name_indexes(local_vertex_indexes)

        graph.nearby_centroid_map = [[local_vertex_indexes[v] for v in vertex] for vertex in graph.nearby_centroid_map]

        return graph
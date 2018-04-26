import numpy as np


class VertexSubsampler:

    inner = None
    negative_sample_rate = None

    max_edges = 10000
    max_edges_per_vertex = 2000

    def __init__(self, inner, negative_sample_rate):
        self.inner = inner
        self.negative_sample_rate = negative_sample_rate

    def extend(self, example):
        example = self.inner.extend(example)

        negative_sample_rate = min(self.negative_sample_rate, example.graph.count_entity_vertices())
        golds = example.get_gold_indexes()
        centroids = example.get_centroid_indexes()
        potential_negatives = np.random.choice(example.get_entity_vertex_indexes(), negative_sample_rate, replace=False)
        potential_negatives = np.setdiff1d(potential_negatives, centroids)
        kept_vertices_first_round = np.unique(np.concatenate((golds, potential_negatives))).astype(np.int32)

        kept_edges_first_round = np.logical_or(np.isin(example.graph.edges[:, 0], kept_vertices_first_round),
                                            np.isin(example.graph.edges[:, 2], kept_vertices_first_round))


        kept_edges_count = np.sum(kept_edges_first_round)
        if kept_edges_count > self.max_edges:
            edges_by_vertex = [np.logical_and(np.logical_or(np.isin(example.graph.edges[:, 0], [x]),
                                                            np.isin(example.graph.edges[:, 2], [x])),
                                              kept_edges_first_round)
                               for x in kept_vertices_first_round]

            kept_edges_first_round = np.zeros((example.graph.edges.shape[0]), dtype=np.bool)

            for kept_edges in edges_by_vertex:
                kept_count = np.sum(kept_edges)
                if kept_count > self.max_edges_per_vertex:
                    filtered = np.random.choice(np.where(kept_edges)[0], self.max_edges_per_vertex, replace=False)
                    kept_edges = np.zeros((example.graph.edges.shape[0]), dtype=np.bool)
                    kept_edges[filtered] = True

                    kept_edges_first_round = np.logical_or(kept_edges_first_round, kept_edges)


        kept_vertices_second_round = np.unique(np.concatenate((example.graph.edges[kept_edges_first_round][:,0], example.graph.edges[kept_edges_first_round][:,2], centroids)))

        kept_edges_second_round = np.logical_and(np.isin(example.graph.edges[:, 0], kept_vertices_second_round),
                                            np.isin(example.graph.edges[:, 2], kept_vertices_second_round))

        vertex_map = {kept_vertex: i for i, kept_vertex in
                      enumerate(np.arange(example.graph.vertices.shape[0], dtype=np.int32)[kept_vertices_second_round])}

        new_label_to_vertex_map = {label:vertex_map[index] for label, index in example.graph.vertex_label_to_index_map.items() if index in vertex_map}

        example.graph.map_name_indexes(vertex_map)

        example.graph.set_label_to_index_map(new_label_to_vertex_map)
        example.graph.vertices = example.graph.vertices[kept_vertices_second_round]
        example.graph.vertex_types = example.graph.vertex_types[kept_vertices_second_round]
        example.graph.edges = example.graph.edges[kept_edges_second_round]
        example.graph.padded_edge_bow_matrix = example.graph.padded_edge_bow_matrix[kept_edges_second_round]

        for i in range(example.graph.edges.shape[0]):
            example.graph.edges[i][0] = vertex_map[example.graph.edges[i][0]]
            example.graph.edges[i][2] = vertex_map[example.graph.edges[i][2]]

        example.index_mentions()
        example.index_gold_answers()

        example.graph.entity_vertex_indexes = np.array([vertex_map[v] for v in example.graph.entity_vertex_indexes if v in vertex_map])

        example.graph.nearby_centroid_map = [example.graph.nearby_centroid_map[kept_vertex] for kept_vertex,i in vertex_map.items()]
        example.graph.vertex_max_scores = np.array([example.graph.vertex_max_scores[kept_vertex] for kept_vertex,i in vertex_map.items()]).astype(np.float32) \
            if example.graph.vertex_max_scores is not None else np.zeros(example.graph.vertices.shape[0], dtype=np.float32)

        return example
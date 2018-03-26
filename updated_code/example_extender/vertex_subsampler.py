import numpy as np


class VertexSubsampler:

    inner = None
    negative_sample_rate = None

    def __init__(self, inner, negative_sample_rate):
        self.inner = inner
        self.negative_sample_rate = negative_sample_rate

    def extend(self, example, mode):
        example = self.inner.extend(example, mode)
        if mode != "train":
            return example

        negative_sample_rate = min(self.negative_sample_rate, example.graph.count_vertices())
        centroids = example.get_centroid_indexes()
        golds = example.get_gold_indexes()
        potential_negatives = np.random.choice(example.graph.vertices.shape[0], negative_sample_rate, replace=False)
        kept_vertices_first_round = np.unique(np.concatenate((golds, potential_negatives))).astype(np.int32)

        kept_edges_first_round = np.logical_or(np.isin(example.graph.edges[:, 0], kept_vertices_first_round),
                                            np.isin(example.graph.edges[:, 2], kept_vertices_first_round))
        kept_vertices_second_round = np.unique(np.concatenate((example.graph.edges[kept_edges_first_round][:,0], example.graph.edges[kept_edges_first_round][:,2])))

        kept_edges_second_round = np.logical_and(np.isin(example.graph.edges[:, 0], kept_vertices_second_round),
                                            np.isin(example.graph.edges[:, 2], kept_vertices_second_round))

        vertex_map = {kept_vertex: i for i,kept_vertex in enumerate(kept_vertices_second_round)}
        new_label_to_vertex_map = {label:vertex_map[index] for label, index in example.graph.vertex_label_to_index_map.items() if index in vertex_map}

        example.graph.vertex_label_to_index_map = new_label_to_vertex_map
        example.graph.vertices = kept_edges_first_round
        example.graph.edges = example.graph.edges[kept_edges_second_round]

        for i in range(example.graph.edges.shape[0]):
            example.graph.edges[i][0] = vertex_map[example.graph.edges[i][0]]
            example.graph.edges[i][2] = vertex_map[example.graph.edges[i][2]]

        return example
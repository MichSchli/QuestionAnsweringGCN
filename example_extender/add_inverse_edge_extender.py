import numpy as np


class AddInverseEdgeExtender:

    relation_index = None
    inner = None

    def __init__(self, inner, relation_index):
        self.inner = inner
        self.relation_index = relation_index

    def extend(self, example):
        example = self.inner.extend(example)
        edges = example.graph.edges

        inverse_edges = np.copy(edges)
        inverse_edges[:,0] = edges[:,2]
        inverse_edges[:,1] += self.relation_index.inverse_edge_delimiter
        inverse_edges[:,2] = edges[:,0]

        example.graph.edges = np.concatenate((edges, inverse_edges))
        example.graph.padded_edge_bow_matrix = np.concatenate((example.graph.padded_edge_bow_matrix, example.graph.padded_edge_bow_matrix))

        return example


from model.hypergraph_model import HypergraphModel
import numpy as np


class SubsampleVerticesService:

    negative_sample_rate = None

    def __init__(self, negative_sample_rate):
        self.negative_sample_rate = negative_sample_rate

    def subsample_vertices(self, graph, positives):
        new_graph = HypergraphModel()
        new_graph.name_edge_type = graph.name_edge_type
        new_graph.type_edge_type = graph.type_edge_type
        new_graph.relation_map = graph.relation_map

        negatives = np.where()

        return new_graph
import numpy as np


class VertexFeatureCombiner:

    batch = None

    def __init__(self, batch):
        self.batch = batch

    def get_max_score_by_vertex(self):
        index_lists = [example.get_vertex_max_scores() for example in self.batch.examples]

        return np.concatenate(index_lists).astype(np.float32)

    def get_combined_vertex_types(self):
        lists = [example.graph.get_vertex_types() for example in self.batch.examples]

        return np.concatenate(lists)
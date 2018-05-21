import tensorflow as tf

class GcnTypeFeatureWrapper:

    inner_feature = None
    graph = None

    def __init__(self, inner_feature, graph, i):
        self.inner_feature = inner_feature
        self.graph = graph
        self.i = i

    def get(self):
        return tf.nn.embedding_lookup(self.inner_feature.get(), self.graph.get_gcn_type_edge_indices(self.i))

    def get_width(self):
        return self.inner_feature.get_width()

    def prepare_variables(self):
        self.inner_feature.prepare_variables()
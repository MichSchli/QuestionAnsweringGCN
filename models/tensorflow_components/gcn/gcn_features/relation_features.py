import tensorflow as tf


class RelationFeatures:

    def __init__(self, graph, width, relation_index, edge_type=None):
        self.graph = graph
        self.width = width
        self.relation_index = relation_index
        self.edge_type = edge_type

    def get(self):
        relation_indices = self.graph.get_edge_types(self.edge_type)

        return tf.nn.embedding_lookup(self.embeddings, relation_indices)

    def get_width(self):
        return self.width

    def prepare_variables(self):
        initializer = self.relation_index.get_all_vectors()
        self.embeddings = tf.Variable(initializer)

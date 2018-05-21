import tensorflow as tf


class RelationPartFeatures:

    def __init__(self, graph, width, relation_part_index, edge_type=None):
        self.graph = graph
        self.width = width
        self.relation_part_index = relation_part_index
        self.edge_type = None

    def get(self):
        relation_indices = self.graph.get_edge_word_bows(edge_type=self.edge_type)
        all_embeddings = tf.nn.embedding_lookup(self.embeddings, relation_indices)

        return tf.reduce_sum(all_embeddings, axis=-2)

    def get_width(self):
        return self.width

    def prepare_variables(self):
        initializer = self.relation_part_index.get_all_vectors()
        self.embeddings = tf.Variable(initializer)

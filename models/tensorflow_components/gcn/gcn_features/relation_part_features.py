import tensorflow as tf


class RelationPartFeatures:

    def __init__(self, graph, width, relation_part_index):
        self.graph = graph
        self.width = width
        self.relation_part_index = relation_part_index

    def get(self):
        relation_indices = self.graph.get_edge_word_bows()
        all_embeddings = tf.nn.embedding_lookup(self.embeddings, relation_indices)

        return tf.reduce_sum(all_embeddings, axis=-2)

    def get_width(self):
        return self.width

    def prepare_variables(self):
        initializer = self.relation_part_index.get_all_vectors()
        self.embeddings = tf.Variable(initializer)

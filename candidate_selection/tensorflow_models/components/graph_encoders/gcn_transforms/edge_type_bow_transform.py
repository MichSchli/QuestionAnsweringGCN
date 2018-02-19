import tensorflow as tf
import numpy as np

class TypeBowTransform:

    def __init__(self, dimension, n_relation_types, hypergraph, gcn_instructions):
        self.dimension = dimension
        self.hypergraph = hypergraph
        self.gcn_instructions = gcn_instructions
        self.n_relation_types = n_relation_types

    def apply(self, vector):
        types = self.hypergraph.get_edge_bows(senders=self.gcn_instructions["sender_tags"],
                                               receivers=self.gcn_instructions["receiver_tags"],
                                               inverse_edges=self.gcn_instructions["invert"])

        type_biases = tf.reduce_sum(tf.nn.embedding_lookup(self.b, types), -2)
        return vector + type_biases

    def prepare_variables(self):
        self.b = tf.Variable(np.random.normal(0, 1, (self.n_relation_types, self.dimension)).astype(np.float32))
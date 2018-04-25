import tensorflow as tf
import numpy as np

class VertexScoreFeatures:

    def __init__(self, graph, target):
        self.graph = graph
        self.target = target

    def get(self):
        sender_indices, receiver_indices = self.graph.get_edges()

        target_indices = sender_indices if self.target == "senders" else receiver_indices

        vertex_max_score_list = tf.expand_dims(self.graph.get_vertex_max_scores(), -1)
        return tf.nn.embedding_lookup(vertex_max_score_list, target_indices)

    def get_width(self):
        return 1

    def prepare_variables(self):
        pass
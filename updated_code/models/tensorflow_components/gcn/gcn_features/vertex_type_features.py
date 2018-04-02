import tensorflow as tf
import numpy as np

class VertexTypeFeatures:

    def __init__(self, graph, target):
        self.graph = graph
        self.target = target

    def get(self):
        sender_indices, receiver_indices = self.graph.get_edges()

        target_indices = sender_indices if self.target == "senders" else receiver_indices

        vertex_type_list = self.graph.get_vertex_types()
        return tf.nn.embedding_lookup(vertex_type_list, target_indices)

    def get_width(self):
        return 5

    def prepare_variables(self):
        pass
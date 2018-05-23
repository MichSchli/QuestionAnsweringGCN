import tensorflow as tf


class VertexFeatures:

    def __init__(self, graph, target, width, edge_type=None):
        self.graph = graph
        self.width = width
        self.target = target
        self.edge_type=edge_type

    def get(self):
        sender_indices, receiver_indices = self.graph.get_edges(self.edge_type)
        embeddings = self.graph.get_embeddings()

        target_indices = sender_indices if self.target == "senders" else receiver_indices

        return tf.nn.embedding_lookup(embeddings, target_indices)

    def get_width(self):
        return self.width

    def prepare_variables(self):
        pass
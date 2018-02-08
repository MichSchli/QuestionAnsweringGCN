import tensorflow as tf


class SenderFeatures:

    def __init__(self, hypergraph, gcn_instructions):
        self.hypergraph = hypergraph
        self.gcn_instructions = gcn_instructions

    def get(self):
        sender_indices, receiver_indices = self.hypergraph.get_edges(senders=self.gcn_instructions["sender_tags"],
                                                                     receivers=self.gcn_instructions["receiver_tags"],
                                                                     inverse_edges=self.gcn_instructions["invert"])

        sender_embeddings = self.hypergraph.get_embeddings(self.gcn_instructions["sender_tags"])
        return tf.nn.embedding_lookup(sender_embeddings, sender_indices)
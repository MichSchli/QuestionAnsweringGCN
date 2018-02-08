import tensorflow as tf


class ExternalBatchFeatures:

    vectors_by_example = None

    def __init__(self, hypergraph, gcn_instructions):
        self.hypergraph = hypergraph
        self.gcn_instructions = gcn_instructions

    def set_batch_features(self, vectors_by_example):
        self.vectors_by_example = vectors_by_example

    def get(self):
        return self.hypergraph.distribute_to_edges(self.vectors_by_example,
                                                   senders=self.gcn_instructions["sender_tags"],
                                                   receivers=self.gcn_instructions["receiver_tags"],
                                                   inverse_edges=self.gcn_instructions["invert"])
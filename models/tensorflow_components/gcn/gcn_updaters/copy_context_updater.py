import tensorflow as tf
import numpy as np


class CopyContextGcnInitializer:

    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

    def initialize(self, mode):
        return None


class CopyContextGcnUpdater:
    def __init__(self, prefix, in_dimension, out_dimension, graph):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.variable_prefix = prefix
        self.graph = graph

    def update(self, new_observations, previous_context):
        self.graph.update_vertex_embeddings(new_observations)

        return None

    def get_regularization(self):
        return 0
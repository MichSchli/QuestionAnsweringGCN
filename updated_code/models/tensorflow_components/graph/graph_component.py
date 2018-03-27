import tensorflow as tf


class GraphComponent:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    def __init__(self):
        self.variables = {}
        self.variables["n_vertices"] = tf.placeholder(tf.int32)
        self.variables["entity_vertex_indexes"] = tf.placeholder(tf.int32)

    def initialize_zero_embeddings(self, dimension):
        self.vertex_embeddings = tf.zeros((self.get_variable("n_vertices"), dimension))

    def get_target_vertex_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("entity_vertex_indexes"))

    def get_variable(self, name):
        return self.variables[name]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["n_vertices"] = batch.count_all_vertices()
        self.variable_assignments["entity_vertex_indexes"] = batch.get_combined_vertex_indexes()
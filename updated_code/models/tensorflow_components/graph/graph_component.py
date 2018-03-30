import tensorflow as tf


class GraphComponent:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    def __init__(self):
        self.variables = {}
        self.variables["n_vertices"] = tf.placeholder(tf.int32)
        self.variables["entity_vertex_indexes"] = tf.placeholder(tf.int32)
        self.variables["sender_indices"] = tf.placeholder(tf.int32)
        self.variables["receiver_indices"] = tf.placeholder(tf.int32)
        self.variables["edge_type_indices"] = tf.placeholder(tf.int32)
        self.variables["edge_bow_matrix"] = tf.placeholder(tf.int32)

    def initialize_zero_embeddings(self, dimension):
        self.vertex_embeddings = tf.zeros((self.get_variable("n_vertices"), dimension))

    def get_embeddings(self):
        return self.vertex_embeddings

    def update_vertex_embeddings(self, updated_embeddings):
        self.vertex_embeddings = updated_embeddings

    def get_target_vertex_embeddings(self):
        return tf.nn.embedding_lookup(self.vertex_embeddings, self.get_variable("entity_vertex_indexes"))

    def get_variable(self, name):
        return self.variables[name]

    def get_edges(self):
        return self.variables["sender_indices"], self.variables["receiver_indices"]

    def get_edge_types(self):
        return self.variables["edge_type_indices"]

    def get_edge_word_bows(self):
        return self.variables["edge_bow_matrix"]

    def count_vertices(self):
        return self.variables["n_vertices"]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["n_vertices"] = batch.count_all_vertices()
        self.variable_assignments["entity_vertex_indexes"] = batch.get_combined_entity_vertex_map_indexes()
        self.variable_assignments["sender_indices"] = batch.get_combined_sender_indices()
        self.variable_assignments["receiver_indices"] = batch.get_combined_receiver_indices()
        self.variable_assignments["edge_type_indices"] = batch.get_combined_edge_type_indices()
        self.variable_assignments["edge_bow_matrix"] = batch.get_padded_edge_part_type_matrix()

    def get_regularization(self):
        return 0
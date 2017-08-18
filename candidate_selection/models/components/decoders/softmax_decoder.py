import tensorflow as tf


class SoftmaxDecoder:

    variables = None

    def __init__(self, tensorflow_variables_holder):
        self.variables = tensorflow_variables_holder

    def decode_to_prediction(self, entity_scores):
        entity_vertex_scores = tf.concat((tf.constant([0], dtype=tf.float32), entity_scores), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores, self.variables.get_variable("vertex_lookup_matrix"))

        def map_function(x):
            vals = tf.nn.softmax(x[0][:x[1]])
            zeros_needed = tf.reduce_max(self.variables.get_variable("vertex_count_per_hypergraph")) - x[1]
            mask = tf.zeros((1, zeros_needed), dtype=tf.float32)
            return tf.concat((tf.expand_dims(vals, 0), mask), 1)

        elems = (entity_vertex_scores_distributed, self.variables.get_variable("vertex_count_per_hypergraph"))
        alternate = tf.map_fn(map_function, elems, dtype=tf.float32)
        return alternate

    def decode_to_loss(self, entity_scores):
        entity_vertex_scores = tf.concat((tf.constant([0], dtype=tf.float32), entity_scores), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores,
                                                                  self.variables.get_variable("vertex_lookup_matrix"))
        gold_scores = self.variables.get_variable("gold_lookup_matrix")

        def map_function(x):
            golds = x[2][:x[1]]
            vals = tf.nn.softmax_cross_entropy_with_logits(logits=x[0][:x[1]], labels=golds)
            return vals

        elems = (entity_vertex_scores_distributed, self.variables.get_variable("vertex_count_per_hypergraph"), gold_scores)
        alternate = tf.map_fn(map_function, elems, dtype=tf.float32)
        return tf.reduce_sum(alternate)

    def prepare_variables(self, mode='predict'):
        self.variables.add_variable("vertex_lookup_matrix", tf.placeholder(tf.int32))
        self.variables.add_variable("vertex_count_per_hypergraph", tf.placeholder(tf.int32))

        if mode == 'train':
            self.variables.add_variable("gold_lookup_matrix", tf.placeholder(tf.float32))

    def handle_variable_assignment(self, vertex_lookup_matrix, vertex_count_per_hypergraph):
        self.variables.assign_variable("vertex_lookup_matrix", vertex_lookup_matrix)
        self.variables.assign_variable("vertex_count_per_hypergraph", vertex_count_per_hypergraph)

    def assign_gold_variable(self, gold_matrix):
        self.variables.assign_variable("gold_lookup_matrix", gold_matrix)

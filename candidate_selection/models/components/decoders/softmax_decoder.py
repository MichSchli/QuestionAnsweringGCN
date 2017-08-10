import tensorflow as tf


class SoftmaxDecoder:

    def __init__(self):
        pass

    def decode_to_prediction(self, entity_scores, vertex_lookup_matrix, vertex_count_per_hypergraph):
        entity_vertex_scores = tf.concat((tf.Variable([0], dtype=tf.float32, trainable=False), entity_scores), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores, vertex_lookup_matrix)

        def map_function(x):
            vals = tf.nn.softmax(x[0][:x[1]])
            zeros_needed = 8 - x[1]
            mask = tf.zeros((1, zeros_needed), dtype=tf.float32)
            return tf.concat((tf.expand_dims(vals, 0), mask), 1)

        elems = (entity_vertex_scores_distributed, vertex_count_per_hypergraph)
        alternate = tf.map_fn(map_function, elems, dtype=tf.float32)
        return alternate
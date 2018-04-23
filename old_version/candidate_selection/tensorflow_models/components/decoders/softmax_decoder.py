import tensorflow as tf
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class SoftmaxDecoder(AbstractComponent):

    variables = None

    def __init__(self, tensorflow_variables_holder, loss_type):
        self.variables = tensorflow_variables_holder
        self.loss_type = loss_type

    def decode_to_prediction(self, entity_scores):
        entity_vertex_scores = tf.concat((tf.constant([0], dtype=tf.float32), entity_scores), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores, self.variables.get_variable("vertex_lookup_matrix"))

        def map_function(x):
            vals = tf.nn.sigmoid(x[0][:x[1]])
            zeros_needed = tf.reduce_max(self.variables.get_variable("vertex_count_per_hypergraph")) - x[1]
            mask = tf.zeros((1, zeros_needed), dtype=tf.float32)
            return tf.concat((tf.expand_dims(vals, 0), mask), 1)

        elems = (entity_vertex_scores_distributed, self.variables.get_variable("vertex_count_per_hypergraph"))
        alternate = tf.map_fn(map_function, elems, dtype=tf.float32)
        return alternate


    def inner_sigmoid(self, logits, labels):
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)

        losses= math_ops.add(
            relu_logits - logits * labels,
            math_ops.log1p(math_ops.exp(neg_abs_logits)))

        #labels = tf.Print(labels, data=[labels], summarize=100, message="labels")
        positive_cond = (labels > zeros)
        negative_losses = array_ops.where(tf.logical_not(positive_cond), losses, zeros)

        positive_losses = array_ops.where(positive_cond, losses, tf.ones_like(logits)*3000)
        #positive_losses = tf.Print(positive_losses, data=[positive_losses], summarize=100, message="losses")

        total_negative_loss = tf.reduce_mean(negative_losses)
        total_positive_loss = tf.reduce_min(positive_losses)

        total_loss = total_negative_loss + total_positive_loss

        return total_loss

    def decode_to_loss(self, entity_scores, sum_examples=True):
        entity_vertex_scores = tf.concat((tf.constant([0], dtype=tf.float32), entity_scores), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores,
                                                                  self.variables.get_variable("vertex_lookup_matrix"))
        gold_scores = self.variables.get_variable("gold_lookup_matrix")
        #gold_scores = tf.Print(gold_scores, [gold_scores], message="gold lookup matrix: ", summarize=25)

        def map_function(x):
            golds = x[2][:x[1]]

            if self.loss_type == "sigmoid":
                vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x[0][:x[1]], labels=golds)
                loss = tf.reduce_mean(vals)
            elif self.loss_type == "weighted_sigmoid":
                pos_weight = tf.to_float(x[1]) / tf.reduce_sum(tf.to_float(golds))
                vals = tf.nn.weighted_cross_entropy_with_logits(logits=x[0][:x[1]], targets=golds,
                                                                pos_weight=pos_weight)
                loss = tf.reduce_mean(vals)
            elif self.loss_type == "argmax_sigmoid":
                loss = self.inner_sigmoid(x[0][:x[1]], golds)

            return loss

        elems = (entity_vertex_scores_distributed, self.variables.get_variable("vertex_count_per_hypergraph"), gold_scores)
        alternate = tf.map_fn(map_function, elems, dtype=tf.float32)


        if sum_examples:
            return tf.reduce_mean(alternate)
        else:
            return alternate

    def prepare_tensorflow_variables(self, mode='predict'):
        self.variables.add_variable("vertex_lookup_matrix", tf.placeholder(tf.int32))
        self.variables.add_variable("vertex_count_per_hypergraph", tf.placeholder(tf.int32))

        if mode == 'train':
            self.variables.add_variable("gold_lookup_matrix", tf.placeholder(tf.float32))

    def handle_variable_assignment(self, batch_dictionary, mode):
        hypergraph = batch_dictionary["neighborhood_input_model"]
        self.assign_train_variables(hypergraph.entity_vertex_matrix, hypergraph.entity_vertex_slices)

        if mode == "train":
            self.assign_test_variables(batch_dictionary["gold_mask"])

    def assign_train_variables(self, vertex_lookup_matrix, vertex_count_per_hypergraph):
        self.variables.assign_variable("vertex_lookup_matrix", vertex_lookup_matrix)
        self.variables.assign_variable("vertex_count_per_hypergraph", vertex_count_per_hypergraph)

    def assign_test_variables(self, gold_matrix):
        self.variables.assign_variable("gold_lookup_matrix", gold_matrix)

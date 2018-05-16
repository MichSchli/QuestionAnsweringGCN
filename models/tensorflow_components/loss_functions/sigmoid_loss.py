import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class SigmoidLoss:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    def __init__(self):
        self.variables = {}
        self.variables["gold_vector"] = tf.placeholder(tf.float32)
        self.variables["loss_normalization"] = tf.placeholder(tf.float32)

    def get_variable(self, name):
        return self.variables[name]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["gold_vector"] = batch.get_gold_vector()
        self.variable_assignments["loss_normalization"] = batch.get_normalization_by_vertex_count(weight_positives=True)

    def compute_prediction(self, entity_scores):
        preds = tf.nn.sigmoid(entity_scores)
        golds = self.get_variable("gold_vector")

        return preds

    def compute_loss(self, entity_scores):
        golds = self.get_variable("gold_vector")

        #losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=entity_scores, labels=golds)
        #normalized_losses = losses * self.get_variable("loss_normalization")

        #return tf.reduce_sum(normalized_losses)

        return self.WEIRD_LOSS_MOVE(entity_scores, golds)

    def WEIRD_LOSS_MOVE(self, logits, labels):
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)

        losses = math_ops.add(
            relu_logits - logits * labels,
            math_ops.log1p(math_ops.exp(neg_abs_logits)))

        # labels = tf.Print(labels, data=[labels], summarize=100, message="labels")
        positive_cond = (labels > zeros)
        negative_losses = array_ops.where(tf.logical_not(positive_cond), losses, zeros)

        positive_losses = array_ops.where(positive_cond, losses, tf.ones_like(logits) * 0)
        # positive_losses = tf.Print(positive_losses, data=[positive_losses], summarize=100, message="losses")

        #positive_losses = tf.Print(positive_losses, [positive_losses], summarize=100)
        #positive_losses = tf.nn.dropout(positive_losses, keep_prob=0.8)

        #negative_losses = tf.Print(negative_losses, [negative_losses], summarize=100)

        #losses = positive_losses + negative_losses
        #normalized_losses = losses * self.get_variable("loss_normalization")
        #total_loss = tf.reduce_sum(normalized_losses)

        total_negative_loss = tf.reduce_mean(negative_losses)
        total_positive_loss = tf.reduce_mean(positive_losses)

        total_loss = total_negative_loss + total_positive_loss

        return total_loss

    def get_regularization(self):
        return 0
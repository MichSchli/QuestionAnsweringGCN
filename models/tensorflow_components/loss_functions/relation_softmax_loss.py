import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class RelationSoftmaxLoss:

    variables = None
    variable_assignments = None

    def __init__(self):
        self.variables = {}
        self.variables["gold_vector"] = tf.placeholder(tf.float32)

    def get_variable(self, name):
        return self.variables[name]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["gold_vector"] = batch.get_relation_class_labels()

    def compute_prediction(self, entity_scores):
        preds = tf.nn.softmax(entity_scores)
        golds = self.get_variable("gold_vector")

        return preds

    def compute_loss(self, entity_scores):
        golds = self.get_variable("gold_vector")
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=entity_scores, labels=golds)

        loss = tf.reduce_mean(losses)
        return loss

    def get_regularization(self):
        return 0
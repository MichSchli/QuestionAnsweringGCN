import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class MaxPredSigmoidLoss:

    vertex_embeddings = None
    variables = None
    variable_assignments = None

    def __init__(self):
        self.variables = {}
        self.variables["gold_vector"] = tf.placeholder(tf.float32)
        self.variables["loss_normalization"] = tf.placeholder(tf.float32)
        self.variables["padded_vertex_matrix"] = tf.placeholder(tf.int32)
        self.variables["padded_gold_matrix"] = tf.placeholder(tf.float32)
        self.variables["vertex_count"] = tf.placeholder(tf.int32)

    def get_variable(self, name):
        return self.variables[name]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}
        self.variable_assignments["gold_vector"] = batch.get_gold_vector()
        self.variable_assignments["loss_normalization"] = batch.get_normalization_by_vertex_count(weight_positives=True)
        self.variable_assignments["padded_vertex_matrix"] = batch.get_padded_entity_indexes() + 1
        self.variable_assignments["padded_gold_matrix"] = batch.get_padded_gold_matrix()
        self.variable_assignments["vertex_count"] = batch.get_entity_counts()

    def compute_prediction(self, entity_scores):
        preds = tf.nn.sigmoid(entity_scores)
        golds = self.get_variable("gold_vector")

        return preds

    def fancy_loss(self, logits, labels):
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)

        losses = math_ops.add(
            relu_logits - logits * labels,
            math_ops.log1p(math_ops.exp(neg_abs_logits)))

        positive_cond = (labels > zeros)
        negative_losses = array_ops.where(tf.logical_not(positive_cond), losses, zeros)

        positive_losses = array_ops.where(positive_cond, losses, tf.ones_like(logits) * 3000)

        #positive_losses = tf.Print(positive_losses, data=[labels], summarize=100, message="labels")
        #positive_losses = tf.Print(positive_losses, data=[logits], summarize=100, message="logits")
        #positive_losses = tf.Print(positive_losses, data=[tf.nn.sigmoid(logits)], summarize=100, message="preds")
        #positive_losses = tf.Print(positive_losses, data=[positive_losses], summarize=100, message="pos")
        #positive_losses = tf.Print(positive_losses, data=[negative_losses], summarize=100, message="neg")

        total_negative_loss = tf.reduce_max(negative_losses)
        total_positive_loss = tf.reduce_min(positive_losses)

        total_loss = total_negative_loss + total_positive_loss

        return total_loss

    def compute_loss(self, logits):
        labels = self.get_variable("padded_gold_matrix")

        entity_vertex_scores = tf.concat((tf.constant([0], dtype=tf.float32), logits), 0)
        entity_vertex_scores_distributed = tf.nn.embedding_lookup(entity_vertex_scores, self.get_variable("padded_vertex_matrix"))

        def map_function(x):
            golds = x[2][:x[1]]

            loss = self.fancy_loss(x[0][:x[1]], golds)

            return loss

        elems = (entity_vertex_scores_distributed, self.get_variable("vertex_count"), labels)
        loss = tf.map_fn(map_function, elems, dtype=tf.float32)

        return tf.reduce_mean(loss)

        return self.fancy_loss(logits, labels)

    def get_regularization(self):
        return 0
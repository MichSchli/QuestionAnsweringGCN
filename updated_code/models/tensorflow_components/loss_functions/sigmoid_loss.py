import tensorflow as tf


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
        preds = tf.Print(preds, [golds], summarize=400, message="golds")
        preds = tf.Print(preds, [preds], summarize=400, message="predictions")


        return preds

    def compute_loss(self, entity_scores):
        golds = self.get_variable("gold_vector")
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=entity_scores, labels=golds)
        normalized_losses = losses * self.get_variable("loss_normalization")

        #normalized_losses = tf.Print(normalized_losses, [golds], summarize=400, message="golds")
        #normalized_losses = tf.Print(normalized_losses, [tf.nn.sigmoid(entity_scores)], summarize=400, message="scores")
        #normalized_losses = tf.Print(normalized_losses, [normalized_losses], summarize=400, message="loss")
        normalized_losses = tf.Print(normalized_losses, [tf.reduce_sum(normalized_losses)], summarize=400, message="total_loss")

        return tf.reduce_sum(normalized_losses)
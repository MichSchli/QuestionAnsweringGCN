from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
import tensorflow as tf


class TensorflowSentenceRepresentation(AbstractComponent):

    variables = None
    variables_prefix = None
    centroid_map = None
    n_centroids = None

    def __init__(self, variables, variable_prefix=""):
        self.variables = variables

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

    def prepare_tensorflow_variables(self, mode="train"):
        self.centroid_map = tf.placeholder(tf.int32, shape=[None], name=self.variable_prefix+"centroid_map")
        self.variables.add_variable(self.variable_prefix + "centroid_map", self.centroid_map)
        self.n_centroids = tf.placeholder(tf.int32, shape=[None], name=self.variable_prefix+"centroid_map")
        self.variables.add_variable(self.variable_prefix + "n_centroids", self.n_centroids)

    def handle_variable_assignment(self, batch_dict, mode):
        sentence = batch_dict["question_sentence_input_model"]
        self.variables.assign_variable(self.variable_prefix + "centroid_map", sentence.centroid_map)
        self.variables.assign_variable(self.variable_prefix + "n_centroids", sentence.n_centroids)
import numpy as np
import tensorflow as tf

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class MultilayerPerceptron(AbstractComponent):

    transforms = None
    variable_prefix = None
    variables = None
    weights = None
    biases = None
    l2_scale = None

    def __init__(self, transforms, variables, variable_prefix="", l2_scale=0.0):
        self.transforms = transforms

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.variables = variables
        self.weights = [None]*(len(transforms)-1)
        self.biases = [None]*(len(transforms)-1)
        self.l2_scale=l2_scale

    def prepare_tensorflow_variables(self, mode="train"):
        for i in range(len(self.transforms)-1):
            dim_1 = self.transforms[i]
            dim_2 = self.transforms[i+1]

            glorot_variance = np.sqrt(6)/np.sqrt(dim_1 + dim_2)
            weight_initializer = np.random.uniform(-glorot_variance, glorot_variance, size=(dim_1, dim_2)).astype(np.float32)
            bias_initializer = np.zeros(dim_2, dtype=np.float32)

            self.weights[i] = tf.Variable(weight_initializer, name=self.variable_prefix + "_W" + str(i))
            self.biases[i] = tf.Variable(bias_initializer, name=self.variable_prefix + "_b" + str(i))

    def transform(self, vectors):
        for i in range(len(self.transforms)-1):
            vectors = tf.matmul(vectors, self.weights[i]) + self.biases[i]
            if i < len(self.transforms) - 2:
                vectors = tf.nn.relu(vectors)

        return vectors

    def get_regularization_term(self):
        return self.l2_scale * tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.weights])

    def handle_variable_assignment(self, batch, mode):
        pass
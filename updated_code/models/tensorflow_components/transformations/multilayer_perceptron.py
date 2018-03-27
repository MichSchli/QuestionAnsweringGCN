import numpy as np
import tensorflow as tf

class MultilayerPerceptronComponent:

    transforms = None
    variable_prefix = None
    variables = None
    weights = None
    biases = None
    l2_scale = None
    dropout_rate=None

    def __init__(self, transforms, variable_prefix, l2_scale=0.0, dropout_rate=0.0):
        self.variables = {}

        self.transforms = transforms
        self.weights = [None]*(len(transforms)-1)
        self.biases = [None]*(len(transforms)-1)
        self.l2_scale=l2_scale
        self.dropout_rate=dropout_rate
        self.variable_prefix = variable_prefix

        self.prepare_tensorflow_variables()

    def prepare_tensorflow_variables(self):
        for i in range(len(self.transforms)-1):
            dim_1 = self.transforms[i]
            dim_2 = self.transforms[i+1]

            glorot_variance = np.sqrt(6)/np.sqrt(dim_1 + dim_2)
            weight_initializer = np.random.uniform(-glorot_variance, glorot_variance, size=(dim_1, dim_2)).astype(np.float32)
            bias_initializer = np.zeros(dim_2, dtype=np.float32)

            self.weights[i] = tf.Variable(weight_initializer, name=self.variable_prefix + "_W" + str(i))
            self.biases[i] = tf.Variable(bias_initializer, name=self.variable_prefix + "_b" + str(i))

    def transform(self, vectors, mode):
        for i in range(len(self.transforms)-1):
            if mode == "train" and self.dropout_rate > 0:
                vectors = tf.nn.dropout(vectors, 1-self.dropout_rate)
            vectors = tf.matmul(vectors, self.weights[i]) + self.biases[i]
            if i < len(self.transforms) - 2:
                vectors = tf.nn.relu(vectors)

        return vectors

    def get_regularization_term(self):
        return self.l2_scale * tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.weights])

    def handle_variable_assignment(self, batch, mode):
        pass
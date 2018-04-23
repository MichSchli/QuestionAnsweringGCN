import tensorflow as tf
import numpy as np

class AffineGcnTransform:

    def __init__(self, in_dimension, out_dimension):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

    def apply(self, vector):
        return tf.matmul(vector, self.W) + self.b

    def prepare_variables(self):
        initializer = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(np.float32)
        self.W = tf.Variable(initializer)
        self.b = tf.Variable(np.zeros(self.out_dimension).astype(np.float32))
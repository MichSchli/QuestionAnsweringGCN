import tensorflow as tf


class GcnGates:

    def __init__(self, features, transform, l1_scale=0.0):
        self.features = features
        self.transform = transform
        self.l1_scale = 0.0

        self.prepare_variables()

    def get_gates(self, mode):
        gate_features = tf.concat([f.get() for f in self.features], -1)
        transformed_gates = self.transform.transform(gate_features, mode)
        gates = tf.nn.sigmoid(transformed_gates)

        self.gate_sum = tf.reduce_sum(gates)

        return gates

    def prepare_variables(self):
        [f.prepare_variables() for f in self.features]

    def get_regularization(self):
        return self.transform.get_regularization() + self.l1_scale * self.gate_sum
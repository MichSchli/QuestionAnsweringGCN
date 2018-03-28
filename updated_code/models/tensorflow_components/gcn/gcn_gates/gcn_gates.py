import tensorflow as tf


class GcnGates:

    def __init__(self, features, transform):
        self.features = features
        self.transform = transform

        self.prepare_variables()

    def get_gates(self, mode):
        gate_features = tf.concat([f.get() for f in self.features], -1)
        transformed_gates = self.transform.transform(gate_features, mode)
        gates = tf.nn.sigmoid(transformed_gates)

        return gates

    def prepare_variables(self):
        [f.prepare_variables() for f in self.features]

    def get_regularization_term(self):
        return 0.0001 * self.gate_sum
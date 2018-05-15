import tensorflow as tf


class GcnBiasAndFeatureGates:

    def __init__(self, biases, features, transform_1, transform_2, l1_scale=0.0):
        self.features = features
        self.biases = biases
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.l1_scale = 0.0

        self.prepare_variables()

    def get_gates(self, mode):
        gate_features = tf.concat([f.get() for f in self.features], -1)
        transformed_gates = self.transform_1.transform(gate_features, mode)

        for b in self.biases:
            transformed_gates += b.get()
        transformed_gates = tf.nn.relu(transformed_gates)

        transformed_gates = self.transform_2.transform(transformed_gates, mode)
        gates = tf.nn.sigmoid(transformed_gates)

        self.gate_sum = tf.reduce_sum(gates)

        return gates

    def prepare_variables(self):
        [f.prepare_variables() for f in self.features]
        [b.prepare_variables() for b in self.biases]

    def get_regularization(self):
        return self.transform_1.get_regularization() + self.l1_scale * self.gate_sum + self.transform_2.get_regularization()
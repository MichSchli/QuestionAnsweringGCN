import tensorflow as tf


class GcnGates:

    def __init__(self, features, transforms):
        self.features = features
        self.transforms = transforms

    gates = None

    def get_gates(self):
        if self.gates is not None:
            return self.gates

        gate_features = tf.concat([f.get() for f in self.features], -1)
        transformed_gates = gate_features
        for transform in self.transforms:
            transformed_gates = transform.apply(transformed_gates)

        gates = tf.nn.sigmoid(transformed_gates)
        self.gate_sum = tf.reduce_sum(gates)

        self.gates = gates

        return gates

    def prepare_variables(self):
        [t.prepare_variables() for t in self.transforms]

    def get_regularization_term(self):
        return 0.0001 * self.gate_sum
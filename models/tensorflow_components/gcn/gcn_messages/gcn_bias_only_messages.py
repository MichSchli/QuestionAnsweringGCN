import tensorflow as tf


class GcnBiasOnlyMessages:

    biases = None

    def __init__(self, biases):
        self.biases = biases
        self.prepare_variables()

    def get_messages(self, mode):
        biases = 0
        for b in self.biases:
            biases += b.get()

        return biases

    def prepare_variables(self):
        [f.prepare_variables() for f in self.biases]

    def get_regularization(self):
        return 0
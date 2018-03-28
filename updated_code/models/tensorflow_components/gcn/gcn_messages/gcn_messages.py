import tensorflow as tf


class GcnMessages:

    def __init__(self, features, transform):
        self.features = features
        self.transform = transform
        self.prepare_variables()

    def get_messages(self, mode):
        if len(self.features) > 0:
            message_features = tf.concat([f.get() for f in self.features], -1)
        else:
            message_features = 0
        transformed_messages = self.transform.transform(message_features, mode)

        return transformed_messages

    def prepare_variables(self):
        [f.prepare_variables() for f in self.features]
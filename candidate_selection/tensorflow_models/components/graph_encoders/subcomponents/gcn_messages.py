import tensorflow as tf
import numpy as np

class GcnMessages:

    sender_tags = None
    receiver_tags = None
    propagate_on_inverse_edges = None

    def __init__(self, features, transforms):
        self.features = features
        self.transforms = transforms

    def get_messages(self):
        if len(self.features) > 0:
            message_features = tf.concat([f.get() for f in self.features], -1)
        else:
            message_features = 0
        transformed_messages = message_features
        for transform in self.transforms:
            transformed_messages = transform.apply(transformed_messages)

        return transformed_messages

    def prepare_variables(self):
        [t.prepare_variables() for t in self.transforms]
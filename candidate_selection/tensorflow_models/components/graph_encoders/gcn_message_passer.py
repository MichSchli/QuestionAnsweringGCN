import tensorflow as tf
import numpy as np

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.external_batch_features import \
    ExternalBatchFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.sender_features import SenderFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.affine_transform import \
    AffineGcnTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.relu_transform import ReluTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.type_bias_transform import \
    TypeBiasTransform
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_gates import GcnGates
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_messages import GcnMessages


class GcnConcatMessagePasser(AbstractComponent):


    senders = None
    receivers = None
    use_inverse_edges_instead = None

    def __init__(self, hypergraph, senders="events", receivers="entities", inverse_edges=False, gate_mode="none"):
        self.hypergraph = hypergraph

        self.senders = senders
        self.receivers = receivers
        self.use_inverse_edges_instead = inverse_edges
        if gate_mode != "none":
            self.use_gates = True
        else:
            self.use_gates = False
        self.gate_mode = gate_mode

    def set_gate_key(self, gate_key):
        self.sentence_features.set_batch_features(gate_key)

    def get_regularization_term(self):
        if self.use_gates:
            return self.gates.get_regularization_term()

    def get_update(self, hypergraph):
        sender_indices, receiver_indices = hypergraph.get_edges(senders=self.senders, receivers=self.receivers, inverse_edges=self.use_inverse_edges_instead)
        receiver_embeddings = hypergraph.get_embeddings(self.receivers)
        incidence_matrix = self.get_unnormalized_incidence_matrix(receiver_indices, tf.shape(receiver_embeddings)[0])
        messages = self.messages.get_messages()

        if self.use_gates:
            gates = self.gates.get_gates()

        if self.use_gates:
            messages = messages * gates

        sent_messages = tf.sparse_tensor_dense_matmul(incidence_matrix, messages)
        return sent_messages

    def get_unnormalized_incidence_matrix(self, receiver_indices, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([receiver_indices, message_indices])))
        mtr_shape = tf.to_int64(tf.stack([number_of_receivers, message_count]))

        tensor = tf.SparseTensor(indices=mtr_indices,
                                 values=mtr_values,
                                 dense_shape=mtr_shape)

        return tensor

    def get_globally_normalized_incidence_matrix(self, receiver_indices, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([receiver_indices, message_indices])))
        mtr_shape = tf.to_int64(tf.stack([number_of_receivers, message_count]))

        tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                 values=mtr_values,
                                 dense_shape=mtr_shape))

        return tensor

    def prepare_variables(self):
        if self.use_gates:
            self.gates.prepare_variables()

        self.messages.prepare_variables()
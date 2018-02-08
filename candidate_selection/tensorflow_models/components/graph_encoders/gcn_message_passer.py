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

    facts = None
    variables = None
    random = None
    dimension = None
    variable_prefix = None

    n_coefficients = 1
    submatrix_d = None

    senders = None
    receivers = None
    use_inverse_edges_instead = None

    def __init__(self, hypergraph, facts, variables, dimension, variable_prefix="", senders="events", receivers="entities", inverse_edges=False, weights="block", biases="constant", gate_mode="none", gate_input_dim=1):
        self.facts = facts
        self.variables = variables
        self.dimension = dimension
        self.submatrix_d = int(dimension / self.n_coefficients)
        self.weight_type = weights
        self.bias_type = biases
        self.hypergraph = hypergraph

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.senders = senders
        self.receivers = receivers
        self.use_inverse_edges_instead = inverse_edges
        if gate_mode != "none":
            self.use_gates = True
        else:
            self.use_gates = False
        self.gate_mode = gate_mode
        self.gate_input_dim = gate_input_dim

        gcn_instructions = {"sender_tags": self.senders,
                            "receiver_tags": self.receivers,
                            "invert": self.use_inverse_edges_instead}

        self.sentence_features = ExternalBatchFeatures(self.hypergraph, gcn_instructions)
        message_features = [SenderFeatures(self.hypergraph, gcn_instructions),
                            self.sentence_features]
        message_transforms = [AffineGcnTransform(self.dimension*2, self.dimension),
                              TypeBiasTransform(self.dimension, self.facts.number_of_relation_types, self.hypergraph, gcn_instructions),
                              ReluTransform(self.dimension)]

        gate_features = [SenderFeatures(self.hypergraph, gcn_instructions),
                         self.sentence_features]
        gate_transforms = [AffineGcnTransform(self.dimension*2, self.dimension),
                           TypeBiasTransform(self.dimension, self.facts.number_of_relation_types, self.hypergraph, gcn_instructions),
                           ReluTransform(self.dimension),
                           AffineGcnTransform(self.dimension, 1)]

        self.messages = GcnMessages(message_features,
                                    message_transforms)
        self.gates = GcnGates(gate_features,
                              gate_transforms)

    def set_gate_features(self, features):
        self.gate_features = features

    def set_gate_key(self, gate_key):
        self.sentence_features.set_batch_features(gate_key)

    def get_regularization_term(self):
        if self.use_gates:
            return self.gates.get_regularization_term()

    def distribute_to_edges(self, hypergraph, vectors_by_sentences):
        return hypergraph.distribute_to_edges(vectors_by_sentences, senders=self.senders, receivers=self.receivers, inverse_edges=self.use_inverse_edges_instead)

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

    def get_locally_normalized_incidence_matrix(self, receiver_indices, message_types, number_of_receivers):
        mtr_values = tf.to_float(tf.ones_like(receiver_indices))

        message_count = tf.shape(receiver_indices)[0]
        message_indices = tf.range(message_count, dtype=tf.int32)

        mtr_indices = tf.to_int64(tf.transpose(tf.stack([message_types, receiver_indices, message_indices])))

        mtr_shape = tf.to_int64(tf.stack([self.facts.number_of_relation_types, number_of_receivers, message_count]))

        tensor = tf.sparse_softmax(tf.SparseTensor(indices=mtr_indices,
                                                   values=mtr_values,
                                                   dense_shape=mtr_shape))

        return tf.sparse_reduce_sum_sparse(tensor, 0)

    def prepare_variables(self):
        if self.use_gates:
            self.gates.prepare_variables()
        if False: #self.use_gates:
            if self.gate_mode == "features_given":
                n_gate_features = self.gate_input_dim
                initializer = np.random.normal(0, 0.01, size=(n_gate_features, self.dimension)).astype(np.float32)
                self.gate_transform = tf.Variable(initializer, name=self.variable_prefix + "gate_weights")
                self.gate_bias = tf.Variable(np.zeros(self.dimension).astype(np.float32))
            elif self.gate_mode == "type_key_comparison":
                initializer = np.random.normal(0, 0.01, size=(self.facts.number_of_relation_types, self.dimension)).astype(np.float32) * 0
                self.gate_type_embeddings = tf.Variable(initializer, name=self.variable_prefix + "gate_type_features")
                n_gate_features = self.dimension * 2

                initializer2 = np.random.normal(0, 0.01, size=(n_gate_features, self.dimension)).astype(np.float32)
                self.gate_transform = tf.Variable(initializer2, name=self.variable_prefix + "gate_weights_1")
                self.gate_bias = tf.Variable(np.zeros(self.dimension).astype(np.float32))

                initializer3 = np.random.normal(0, 0.01, size=self.dimension).astype(np.float32)
                self.gate_transform_2 = tf.Variable(initializer3, name=self.variable_prefix + "gate_weights_2")
                self.gate_bias_2 = tf.Variable(np.zeros(1).astype(np.float32))

        self.messages.prepare_variables()
from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnMessagePasser
import numpy as np
import tensorflow as tf


class NormalGcnPropagationUnit(AbstractComponent):

    self_weight_type = None
    self_bias_type = None

    def __init__(self, prefix, number_of_relation_types, variables, in_dimension, out_dimension, hypergraph,
                 weights="block", biases="constant", self_weight="full", self_bias="constant", add_inverse_relations=True, gate_mode="none", gate_input_dim=1):
        self.add_inverse_relations = add_inverse_relations
        self.hypergraph = hypergraph
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.variable_prefix = prefix

        self.self_weight_type = self_weight
        self.self_bias_type = self_bias

    def get_optimizable_parameters(self):
        params = [self.W_self_entities, self.W_self_events]

        params += self.gcn_encoder_ev_to_en.get_optimizable_parameters()
        params += self.gcn_encoder_en_to_ev.get_optimizable_parameters()
        params += self.gcn_encoder_en_to_en.get_optimizable_parameters()
        params += self.gcn_encoder_ev_to_en_invert.get_optimizable_parameters()
        params += self.gcn_encoder_en_to_ev_invert.get_optimizable_parameters()
        params += self.gcn_encoder_en_to_en_invert.get_optimizable_parameters()

        return params

    def handle_variable_assignment(self, batch_dict, mode):
        pass

    def get_regularization_term(self):
        reg = self.gcn_encoder_ev_to_en.get_regularization_term()
        reg += self.gcn_encoder_en_to_ev.get_regularization_term()
        reg += self.gcn_encoder_en_to_en.get_regularization_term()
        if self.add_inverse_relations:
            reg += self.gcn_encoder_ev_to_en_invert.get_regularization_term()
            reg += self.gcn_encoder_en_to_ev_invert.get_regularization_term()
            reg += self.gcn_encoder_en_to_en_invert.get_regularization_term()

        return reg

    def set_gate_features(self, features, type):
        if type == "entities":
            self.gcn_encoder_en_to_ev.set_gate_features(features)
            self.gcn_encoder_en_to_en.set_gate_features(features)

            if self.add_inverse_relations:
                self.gcn_encoder_en_to_ev_invert.set_gate_features(features)
                self.gcn_encoder_en_to_en_invert.set_gate_features(features)
        elif type == "events":
            self.gcn_encoder_ev_to_en.set_gate_features(features)

            if self.add_inverse_relations:
                self.gcn_encoder_ev_to_en_invert.set_gate_features(features)

    def set_gate_key(self, key):
        self.gcn_encoder_en_to_ev.set_gate_key(key)
        self.gcn_encoder_en_to_en.set_gate_key(key)
        self.gcn_encoder_ev_to_en.set_gate_key(key)

        if self.add_inverse_relations:
            self.gcn_encoder_en_to_ev_invert.set_gate_key(key)
            self.gcn_encoder_en_to_en_invert.set_gate_key(key)
            self.gcn_encoder_ev_to_en_invert.set_gate_key(key)


    def prepare_tensorflow_variables(self, mode="train"):
        self.gcn_encoder_ev_to_en.prepare_variables()
        self.gcn_encoder_en_to_ev.prepare_variables()
        self.gcn_encoder_en_to_en.prepare_variables()

        if self.add_inverse_relations:
            self.gcn_encoder_ev_to_en_invert.prepare_variables()
            self.gcn_encoder_en_to_ev_invert.prepare_variables()
            self.gcn_encoder_en_to_en_invert.prepare_variables()

        initializer_event_weight = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_events = tf.Variable(initializer_event_weight, name=self.variable_prefix + "event_transform_weights")
        self.b_events = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "event_transform_bias")

        initializer_event_weight_2 = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
            np.float32)
        self.W_events_2 = tf.Variable(initializer_event_weight_2, name=self.variable_prefix + "event_transform_weights_2")
        self.b_events_2 = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "event_transform_bias_2")

        initializer_v = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)
        self.W_self_entities = tf.Variable(initializer_v, name=self.variable_prefix + "self_entitity_weights")

        initializer_e = np.random.normal(0, 0.01, size=(self.in_dimension, self.out_dimension)).astype(
                np.float32)

        self.W_self_events = tf.Variable(initializer_e, name=self.variable_prefix + "self_event_weights")

        self.b_self_entities = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_entitity_bias")
        self.b_self_events = tf.Variable(np.zeros(self.out_dimension).astype(np.float32), name=self.variable_prefix + "self_event_bias")

    def get_edge_gates(self):
        edge_gates = [None]*6

        edge_gates[0] = self.gcn_encoder_en_to_ev.get_edge_gates()
        edge_gates[1] = self.gcn_encoder_ev_to_en.get_edge_gates()
        edge_gates[2] = self.gcn_encoder_en_to_en.get_edge_gates()
        edge_gates[3] = self.gcn_encoder_en_to_ev_invert.get_edge_gates()
        edge_gates[4] = self.gcn_encoder_ev_to_en_invert.get_edge_gates()
        edge_gates[5] = self.gcn_encoder_en_to_en_invert.get_edge_gates()

        return edge_gates

    def propagate(self):
        # Propagate information to events:
        # For now apply no self transform to events
        event_self_loop_messages = tf.matmul(self.hypergraph.event_vertex_embeddings, self.W_self_events)

        event_self_loop_messages += self.b_self_events
        event_self_loop_messages = tf.nn.relu(event_self_loop_messages)

        self_loop_messages = tf.matmul(self.hypergraph.entity_vertex_embeddings,
                                                                  self.W_self_entities)

        self_loop_messages += self.b_self_entities
        self_loop_messages = tf.nn.relu(self_loop_messages)

        event_vertex_embeddings = self.gcn_encoder_en_to_ev.get_update(self.hypergraph)
        if self.add_inverse_relations:
            event_vertex_embeddings += self.gcn_encoder_en_to_ev_invert.get_update(self.hypergraph)

        entity_vertex_embeddings = self.gcn_encoder_ev_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_ev_to_en_invert.get_update(self.hypergraph)

        entity_vertex_embeddings += self.gcn_encoder_en_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_en_to_en_invert.get_update(self.hypergraph)

        self.hypergraph.entity_vertex_embeddings = entity_vertex_embeddings + self_loop_messages
        self.hypergraph.event_vertex_embeddings = event_vertex_embeddings + event_self_loop_messages

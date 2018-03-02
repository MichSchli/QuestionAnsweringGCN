from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnMessagePasser
import numpy as np
import tensorflow as tf

from candidate_selection.tensorflow_models.components.graph_encoders.self_loops.cell_update import CellUpdate
from candidate_selection.tensorflow_models.components.graph_encoders.self_loops.highway_update import HighwayUpdate


class NormalGcnPropagationUnit(AbstractComponent):

    self_weight_type = None
    self_bias_type = None

    def __init__(self, prefix, in_dimension, out_dimension, hypergraph, add_inverse_relations=True):
        self.add_inverse_relations = add_inverse_relations
        self.hypergraph = hypergraph
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.variable_prefix = prefix

        self.entity_self_loop = CellUpdate(prefix + "_entity_self_loop", in_dimension, out_dimension)
        self.event_self_loop = CellUpdate(prefix + "_event_self_loop", in_dimension, out_dimension)


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

        self.entity_self_loop.prepare_tensorflow_variables()
        self.event_self_loop.prepare_tensorflow_variables()

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

        event_vertex_embeddings = self.gcn_encoder_en_to_ev.get_update(self.hypergraph)
        if self.add_inverse_relations:
            event_vertex_embeddings += self.gcn_encoder_en_to_ev_invert.get_update(self.hypergraph)

        entity_vertex_embeddings = self.gcn_encoder_ev_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_ev_to_en_invert.get_update(self.hypergraph)

        entity_vertex_embeddings += self.gcn_encoder_en_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_en_to_en_invert.get_update(self.hypergraph)

        return entity_vertex_embeddings, event_vertex_embeddings

        previous_event_cell_state = self.hypergraph.event_cell_state
        previous_entity_cell_state = self.hypergraph.entity_cell_state

        event_self_loop_messages, event_cell_update = self.event_self_loop.get_update(self.hypergraph.event_vertex_embeddings, event_vertex_embeddings, previous_event_cell_state)
        entity_self_loop_messages, entity_cell_update = self.entity_self_loop.get_update(self.hypergraph.entity_vertex_embeddings, entity_vertex_embeddings, previous_entity_cell_state)

        self.hypergraph.event_cell_state = event_cell_update
        self.hypergraph.entity_cell_state = entity_cell_update

        self.hypergraph.entity_vertex_embeddings = entity_vertex_embeddings + entity_self_loop_messages
        self.hypergraph.event_vertex_embeddings = event_vertex_embeddings + event_self_loop_messages

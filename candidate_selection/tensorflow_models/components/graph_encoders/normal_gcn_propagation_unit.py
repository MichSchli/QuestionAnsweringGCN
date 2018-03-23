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

    def get_regularization_term(self):
        reg = self.gcn_encoder_ev_to_en.get_regularization_term()
        reg += self.gcn_encoder_en_to_ev.get_regularization_term()
        reg += self.gcn_encoder_en_to_en.get_regularization_term()
        if self.add_inverse_relations:
            reg += self.gcn_encoder_ev_to_en_invert.get_regularization_term()
            reg += self.gcn_encoder_en_to_ev_invert.get_regularization_term()
            reg += self.gcn_encoder_en_to_en_invert.get_regularization_term()

        if self.word_to_ev:
            reg += self.gcn_encoder_word_to_ev.get_regularization_term()
            reg += self.gcn_encoder_word_to_ev_invert.get_regularization_term()

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

        if self.word_to_ev:
            self.gcn_encoder_word_to_ev.set_gate_key(key)
            self.gcn_encoder_word_to_ev_invert.set_gate_key(key)

    word_to_ev = False

    def add_word_to_event_encoder(self, encoder, invert=False):
        if not invert:
            self.gcn_encoder_word_to_ev = encoder
        else:
            self.gcn_encoder_word_to_ev_invert = encoder

        self.word_to_ev = True

    def prepare_tensorflow_variables(self, mode="train"):
        self.gcn_encoder_ev_to_en.prepare_variables()
        self.gcn_encoder_en_to_ev.prepare_variables()
        self.gcn_encoder_en_to_en.prepare_variables()

        if self.add_inverse_relations:
            self.gcn_encoder_ev_to_en_invert.prepare_variables()
            self.gcn_encoder_en_to_ev_invert.prepare_variables()
            self.gcn_encoder_en_to_en_invert.prepare_variables()

        if self.word_to_ev:
            self.gcn_encoder_word_to_ev.prepare_variables()
            self.gcn_encoder_word_to_ev_invert.prepare_variables()

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

        if self.word_to_ev:
            event_vertex_embeddings += self.gcn_encoder_word_to_ev.get_update(self.hypergraph)

            word_vertex_embeddings = self.gcn_encoder_word_to_ev_invert.get_update(self.hypergraph)

        entity_vertex_embeddings = self.gcn_encoder_ev_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_ev_to_en_invert.get_update(self.hypergraph)

        entity_vertex_embeddings += self.gcn_encoder_en_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_en_to_en_invert.get_update(self.hypergraph)

        return entity_vertex_embeddings, event_vertex_embeddings, word_vertex_embeddings

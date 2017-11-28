from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser
import numpy as np
import tensorflow as tf


class HypergraphGcnPropagationUnit(AbstractComponent):

    self_weight_type = None
    self_bias_type = None

    def __init__(self, prefix, facts, variables, dimension, hypergraph,
                 weights="block", biases="constant", self_weight="full", self_bias="constant", add_inverse_relations=True):
        self.add_inverse_relations = add_inverse_relations
        self.gcn_encoder_ev_to_en = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_ev_to_en", senders="events",
                                                           receivers="entities",
                                                           weights=weights,
                                                           biases=biases)
        self.gcn_encoder_en_to_ev = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_en_to_ev", senders="entities",
                                                           receivers="events",
                                                           weights=weights,
                                                           biases=biases)
        self.gcn_encoder_en_to_en = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_en_to_en", senders="entities",
                                                           receivers="entities",
                                                           weights=weights,
                                                           biases=biases)
        if add_inverse_relations:
            self.gcn_encoder_ev_to_en_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_ev_to_en", senders="events",
                                                                  receivers="entities", inverse_edges=True,
                                                           weights=weights,
                                                           biases=biases)
            self.gcn_encoder_en_to_ev_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_en_to_ev", senders="entities",
                                                                  receivers="events", inverse_edges=True,
                                                           weights=weights,
                                                           biases=biases)
            self.gcn_encoder_en_to_en_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_en_to_en", senders="entities",
                                                                  receivers="entities", inverse_edges=True,
                                                           weights=weights,
                                                           biases=biases)

        self.hypergraph = hypergraph
        self.facts = facts
        self.dimension = dimension
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

    def prepare_tensorflow_variables(self, mode="train"):
        self.gcn_encoder_ev_to_en.prepare_variables()
        self.gcn_encoder_en_to_ev.prepare_variables()
        self.gcn_encoder_en_to_en.prepare_variables()

        if self.add_inverse_relations:
            self.gcn_encoder_ev_to_en_invert.prepare_variables()
            self.gcn_encoder_en_to_ev_invert.prepare_variables()
            self.gcn_encoder_en_to_en_invert.prepare_variables()

        if self.self_weight_type == "full":
            initializer_v = np.random.normal(0, 0.01, size=(self.dimension, self.dimension)).astype(
                np.float32)

            initializer_e = np.random.normal(0, 0.01, size=(self.dimension, self.dimension)).astype(
                np.float32)

            self.W_self_entities = tf.Variable(initializer_v, name=self.variable_prefix + "self_entitity_weights")
            self.W_self_events = tf.Variable(initializer_e, name=self.variable_prefix + "self_event_weights")

        if self.self_bias_type == "constant":
            self.b_self_entities = tf.Variable(np.zeros(self.dimension).astype(np.float32), name=self.variable_prefix + "self_entitity_weights")
            self.b_self_events = tf.Variable(np.zeros(self.dimension).astype(np.float32), name=self.variable_prefix + "self_event_weights")

    def propagate(self):
        # Propagate information to events:
        # For now apply no self transform to events
        #if self.self_weight_type == "full":
        #    event_self_loop_messages = tf.matmul(self.hypergraph.event_vertex_embeddings, self.W_self_events)

        #if self.self_bias_type == "constant":
        #    event_self_loop_messages += self.b_self_events

        self.hypergraph.event_vertex_embeddings += self.gcn_encoder_en_to_ev.get_update(self.hypergraph)
        if self.add_inverse_relations:
            self.hypergraph.event_vertex_embeddings += self.gcn_encoder_en_to_ev_invert.get_update(self.hypergraph)

        # Propagate information to vertices:

        if self.self_weight_type == "full":
            self_loop_messages = tf.matmul(self.hypergraph.entity_vertex_embeddings,
                                                                  self.W_self_entities)
        else:
            self_loop_messages = self.hypergraph.entity_vertex_embeddings

        if self.self_bias_type == "constant":
            self_loop_messages += self.b_self_entities

        entity_vertex_embeddings = self.gcn_encoder_ev_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_ev_to_en_invert.get_update(self.hypergraph)

        entity_vertex_embeddings += self.gcn_encoder_en_to_en.get_update(self.hypergraph)
        if self.add_inverse_relations:
            entity_vertex_embeddings += self.gcn_encoder_en_to_en_invert.get_update(self.hypergraph)

        self.hypergraph.entity_vertex_embeddings = entity_vertex_embeddings + self_loop_messages

from candidate_selection.models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser


class HypergraphGcnPropagationUnit:

    def __init__(self, prefix, facts, variables, dimension, hypergraph):
        self.gcn_encoder_ev_to_en = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_ev_to_en", senders="events",
                                                           receivers="entities")
        self.gcn_encoder_en_to_ev = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_en_to_ev", senders="entities",
                                                           receivers="events")
        self.gcn_encoder_en_to_en = GcnConcatMessagePasser(facts, variables, dimension,
                                                           variable_prefix=prefix+"_en_to_en", senders="entities",
                                                           receivers="entities")
        self.gcn_encoder_ev_to_en_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_ev_to_en", senders="events",
                                                                  receivers="entities", inverse_edges=True)
        self.gcn_encoder_en_to_ev_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_en_to_ev", senders="entities",
                                                                  receivers="events", inverse_edges=True)
        self.gcn_encoder_en_to_en_invert = GcnConcatMessagePasser(facts, variables, dimension,
                                                                  variable_prefix=prefix+"_en_to_en", senders="entities",
                                                                  receivers="entities", inverse_edges=True)

        self.hypergraph = hypergraph

    def prepare_variables(self):
        self.gcn_encoder_ev_to_en.prepare_variables()
        self.gcn_encoder_en_to_ev.prepare_variables()
        self.gcn_encoder_en_to_en.prepare_variables()
        self.gcn_encoder_ev_to_en_invert.prepare_variables()
        self.gcn_encoder_en_to_ev_invert.prepare_variables()
        self.gcn_encoder_en_to_en_invert.prepare_variables()

    def propagate(self):
        # Propagate information to events:
        self.hypergraph.event_vertex_embeddings = self.gcn_encoder_en_to_ev.get_update(self.hypergraph) \
                                                  + self.gcn_encoder_en_to_ev_invert.get_update(self.hypergraph)

        # Propagate information to vertices:
        entity_vertex_embeddings = self.gcn_encoder_ev_to_en.get_update(self.hypergraph) \
                                   + self.gcn_encoder_ev_to_en_invert.get_update(self.hypergraph)
        entity_vertex_embeddings += self.gcn_encoder_en_to_en.get_update(self.hypergraph) \
                                    + self.gcn_encoder_en_to_en_invert.get_update(self.hypergraph)
        self.hypergraph.entity_vertex_embeddings = entity_vertex_embeddings

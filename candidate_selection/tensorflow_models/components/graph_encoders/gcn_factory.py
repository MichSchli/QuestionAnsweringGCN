from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit


class GcnFactory:

    def get_type_only_gcn(self):
        pass

    def get_gated_type_only_gcn(self, hypergraph, variables, gcn_settings):
        hypergraph_gcn_propagation_units = [None] * gcn_settings["n_layers"]
        for layer in range(gcn_settings["n_layers"]):
            hypergraph_gcn_propagation_units[layer] = HypergraphGcnPropagationUnit("layer_" + str(layer),
                                                                                        gcn_settings[
                                                                                            "n_relation_types"],
                                                                                        variables,
                                                                                        gcn_settings[
                                                                                            "embedding_dimension"],
                                                                                        hypergraph,
                                                                                        weights="identity",
                                                                                        biases="relation_specific",
                                                                                        self_weight="identity",
                                                                                        self_bias="zero",
                                                                                        add_inverse_relations=True,
                                                                                        gate_mode="type_key_comparison")

        return hypergraph_gcn_propagation_units

    def get_gcn(self, layers, message_features=["senders", "types"]):
        pass

    def get_gated_gcn(self, layers, message_features=["senders", "types"], gate_features=["senders", "receivers", "types", "external"]):
        pass
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.external_batch_features import \
    ExternalBatchFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.sender_features import SenderFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.affine_transform import \
    AffineGcnTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.relu_transform import ReluTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.type_bias_transform import \
    TypeBiasTransform
from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_gates import GcnGates
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_messages import GcnMessages


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

            message_instructions = {"sender_tags": "events",
                                    "receiver_tags": "entities",
                                    "invert": False}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_ev_to_en = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)
            message_instructions = {"sender_tags": "entities",
                                    "receiver_tags": "entities",
                                    "invert": False}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_en_to_en = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)
            message_instructions = {"sender_tags": "entities",
                                    "receiver_tags": "events",
                                    "invert": False}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_en_to_ev = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)
            message_instructions = {"sender_tags": "events",
                                    "receiver_tags": "entities",
                                    "invert": True}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_ev_to_en_invert = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)
            message_instructions = {"sender_tags": "entities",
                                    "receiver_tags": "entities",
                                    "invert": True}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_en_to_en_invert = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)
            message_instructions = {"sender_tags": "entities",
                                    "receiver_tags": "events",
                                    "invert": True}
            hypergraph_gcn_propagation_units[layer].gcn_encoder_en_to_ev_invert = self.get_gated_message_passer(gcn_settings,
                                                                                                         hypergraph,
                                                                                                         layer,
                                                                                                         variables,
                                                                                                         message_instructions)

        return hypergraph_gcn_propagation_units

    def get_gcn(self, layers, message_features=["senders", "types"]):
        pass

    def get_gated_gcn(self, layers, message_features=["senders", "types"], gate_features=["senders", "receivers", "types", "external"]):
        pass

    '''
    Message passers:
    '''

    def get_gated_message_passer(self, gcn_settings, hypergraph, layer, variables, message_instructions):
        dimension = gcn_settings["embedding_dimension"]
        number_of_relation_types = gcn_settings["n_relation_types"]
        message_passer = GcnConcatMessagePasser(hypergraph,
                                      senders=message_instructions["sender_tags"],
                                      receivers=message_instructions["receiver_tags"],
                                      gate_mode="type_key_comparison",
                                                inverse_edges=message_instructions["invert"])

        message_passer.sentence_features = ExternalBatchFeatures(hypergraph, message_instructions)
        message_features = [SenderFeatures(hypergraph, message_instructions)]
        message_transforms = [AffineGcnTransform(dimension, dimension),
                              TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                                message_instructions),
                              ReluTransform(dimension)]

        gate_features = [SenderFeatures(hypergraph, message_instructions),
                         message_passer.sentence_features]
        gate_transforms = [AffineGcnTransform(dimension * 2, dimension),
                           TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                           ReluTransform(dimension),
                           AffineGcnTransform(dimension, 1)]

        message_passer.messages = GcnMessages(message_features,
                                    message_transforms)
        message_passer.gates = GcnGates(gate_features,
                              gate_transforms)

        return message_passer
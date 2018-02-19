from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.external_batch_features import \
    ExternalBatchFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.receiver_features import \
    ReceiverFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.sender_features import SenderFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnConcatMessagePasser
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.affine_transform import \
    AffineGcnTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.edge_type_bow_transform import \
    TypeBowTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.relu_transform import ReluTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.type_bias_transform import \
    TypeBiasTransform
from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_gates import GcnGates
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_messages import GcnMessages


class GcnFactory:
    """
    Factory class handling the construction of various GCNs
    """

    def __init__(self):
        pass

    """
    GCNS:
    """

    def get_type_only_gcn(self, hypergraph, variables, gcn_settings):
        pass

    def get_gated_type_only_gcn(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_type_only_message_passer
        hypergraph_gcn_propagation_units = self.make_gcn(gcn_settings, hypergraph, message_passer_function, variables)

        return hypergraph_gcn_propagation_units

    def get_gcn(self, hypergraph, variables, gcn_settings):
        pass

    def get_gated_gcn(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_message_passer
        hypergraph_gcn_propagation_units = self.make_gcn(gcn_settings, hypergraph, message_passer_function, variables)
        return hypergraph_gcn_propagation_units

    def get_gated_gcn_with_relation_bag_features(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_message_passer_with_relation_bag_features
        hypergraph_gcn_propagation_units = self.make_gcn(gcn_settings, hypergraph, message_passer_function, variables)
        return hypergraph_gcn_propagation_units

    """
    Helpers:
    """

    def make_gcn(self, gcn_settings, hypergraph, message_passer_function, variables):
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

            hgpu = hypergraph_gcn_propagation_units[layer]
            self.add_message_passers(gcn_settings, hgpu, hypergraph, message_passer_function, layer)
        return hypergraph_gcn_propagation_units

    def add_message_passers(self, gcn_settings, hgpu, hypergraph, message_passer_function, current_layer):
        message_instructions = {"sender_tags": "events",
                                "receiver_tags": "entities",
                                "invert": False}
        hgpu.gcn_encoder_ev_to_en = message_passer_function(gcn_settings,
                                                            hypergraph,
                                                            message_instructions, current_layer)
        message_instructions = {"sender_tags": "entities",
                                "receiver_tags": "entities",
                                "invert": False}
        hgpu.gcn_encoder_en_to_en = message_passer_function(gcn_settings,
                                                            hypergraph,
                                                            message_instructions, current_layer)
        message_instructions = {"sender_tags": "entities",
                                "receiver_tags": "events",
                                "invert": False}
        hgpu.gcn_encoder_en_to_ev = message_passer_function(gcn_settings,
                                                            hypergraph,
                                                            message_instructions, current_layer)
        message_instructions = {"sender_tags": "events",
                                "receiver_tags": "entities",
                                "invert": True}
        hgpu.gcn_encoder_ev_to_en_invert = message_passer_function(gcn_settings,
                                                                   hypergraph,
                                                                   message_instructions, current_layer)
        message_instructions = {"sender_tags": "entities",
                                "receiver_tags": "entities",
                                "invert": True}
        hgpu.gcn_encoder_en_to_en_invert = message_passer_function(gcn_settings,
                                                                   hypergraph,
                                                                   message_instructions, current_layer)
        message_instructions = {"sender_tags": "entities",
                                "receiver_tags": "events",
                                "invert": True}
        hgpu.gcn_encoder_en_to_ev_invert = message_passer_function(gcn_settings,
                                                                   hypergraph,
                                                                   message_instructions, current_layer)

    """
    Message passers:
    """

    def get_gated_type_only_message_passer(self, gcn_settings, hypergraph, message_instructions):
        dimension = gcn_settings["embedding_dimension"]
        number_of_relation_types = gcn_settings["n_relation_types"]
        message_passer = GcnConcatMessagePasser(hypergraph,
                                                senders=message_instructions["sender_tags"],
                                                receivers=message_instructions["receiver_tags"],
                                                gate_mode="type_key_comparison",
                                                inverse_edges=message_instructions["invert"])

        message_passer.sentence_features = ExternalBatchFeatures(hypergraph, message_instructions)
        message_features = []
        message_transforms = [TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                                message_instructions)]

        gate_features = [message_passer.sentence_features]
        gate_transforms = [AffineGcnTransform(dimension, dimension),
                           TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                           ReluTransform(dimension),
                           AffineGcnTransform(dimension, 1)]

        message_passer.messages = GcnMessages(message_features,
                                              message_transforms)
        message_passer.gates = GcnGates(gate_features,
                                        gate_transforms)

        return message_passer

    def get_gated_message_passer_with_relation_bag_features(self, gcn_settings, hypergraph, message_instructions, current_layer):
        dimension = gcn_settings["embedding_dimension"]
        number_of_relation_types = gcn_settings["n_relation_types"]
        message_passer = GcnConcatMessagePasser(hypergraph,
                                                senders=message_instructions["sender_tags"],
                                                receivers=message_instructions["receiver_tags"],
                                                gate_mode="type_key_comparison",
                                                inverse_edges=message_instructions["invert"])

        if message_instructions["sender_tags"] == "entities" and current_layer == 0:
            input_dim = 1
        else:
            input_dim = dimension

        if message_instructions["receiver_tags"] == "entities" and current_layer == 0:
            input_dim += 1
        else:
            input_dim += dimension

        message_passer.sentence_features = ExternalBatchFeatures(hypergraph, message_instructions)
        message_features = [SenderFeatures(hypergraph, message_instructions),
                            ReceiverFeatures(hypergraph, message_instructions)]
        message_transforms = [AffineGcnTransform(input_dim, dimension),
                              TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                                message_instructions),
                           TypeBowTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                              ReluTransform(dimension)]

        gate_features = [SenderFeatures(hypergraph, message_instructions),
                         ReceiverFeatures(hypergraph, message_instructions),
                         message_passer.sentence_features]
        gate_transforms = [AffineGcnTransform(dimension + input_dim, dimension),
                           TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                           TypeBowTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                           ReluTransform(dimension),
                           AffineGcnTransform(dimension, 1)]

        message_passer.messages = GcnMessages(message_features,
                                              message_transforms)
        message_passer.gates = GcnGates(gate_features,
                                        gate_transforms)

        return message_passer

    def get_gated_message_passer(self, gcn_settings, hypergraph, message_instructions, current_layer):
        dimension = gcn_settings["embedding_dimension"]
        number_of_relation_types = gcn_settings["n_relation_types"]
        message_passer = GcnConcatMessagePasser(hypergraph,
                                                senders=message_instructions["sender_tags"],
                                                receivers=message_instructions["receiver_tags"],
                                                gate_mode="type_key_comparison",
                                                inverse_edges=message_instructions["invert"])

        if message_instructions["sender_tags"] == "entities" and current_layer == 0:
            input_dim = 1
        else:
            input_dim = dimension

        if message_instructions["receiver_tags"] == "entities" and current_layer == 0:
            input_dim += 1
        else:
            input_dim += dimension

        message_passer.sentence_features = ExternalBatchFeatures(hypergraph, message_instructions)
        message_features = [SenderFeatures(hypergraph, message_instructions),
                            ReceiverFeatures(hypergraph, message_instructions)]
        message_transforms = [AffineGcnTransform(input_dim, dimension),
                              TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                                message_instructions),
                              ReluTransform(dimension)]

        gate_features = [SenderFeatures(hypergraph, message_instructions),
                         ReceiverFeatures(hypergraph, message_instructions),
                         message_passer.sentence_features]
        gate_transforms = [AffineGcnTransform(dimension + input_dim, dimension),
                           TypeBiasTransform(dimension, number_of_relation_types, hypergraph,
                                             message_instructions),
                           ReluTransform(dimension),
                           AffineGcnTransform(dimension, 1)]

        message_passer.messages = GcnMessages(message_features,
                                              message_transforms)
        message_passer.gates = GcnGates(gate_features,
                                        gate_transforms)

        return message_passer



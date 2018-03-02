from candidate_selection.tensorflow_models.components.graph_encoders.creation.gcn_feature_factory import \
    GcnFeatureFactory
from candidate_selection.tensorflow_models.components.graph_encoders.creation.gcn_message_passer_factory import \
    GcnMessagePasserFactory
from candidate_selection.tensorflow_models.components.graph_encoders.creation.gcn_transform_factory import \
    GcnTransformFactory
from candidate_selection.tensorflow_models.components.graph_encoders.gcn import GCN
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.external_batch_features import \
    ExternalBatchFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.receiver_features import \
    ReceiverFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.sender_features import SenderFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnMessagePasser
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.affine_transform import \
    AffineGcnTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.edge_type_bow_transform import \
    TypeBowTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.relu_transform import ReluTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.type_bias_transform import \
    TypeBiasTransform
from candidate_selection.tensorflow_models.components.graph_encoders.hypergraph_gcn_propagation_unit import \
    HypergraphGcnPropagationUnit
from candidate_selection.tensorflow_models.components.graph_encoders.normal_gcn_propagation_unit import \
    NormalGcnPropagationUnit
from candidate_selection.tensorflow_models.components.graph_encoders.self_loops.cell_update import CellUpdate
from candidate_selection.tensorflow_models.components.graph_encoders.self_loops.highway_update import HighwayUpdate
from candidate_selection.tensorflow_models.components.graph_encoders.self_loops.residual_update import ResidualUpdate
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_gates import GcnGates
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_messages import GcnMessages


class GcnFactory:
    """
    Factory class handling the construction of various GCNs
    """

    def __init__(self):
        self.feature_factory = GcnFeatureFactory()
        self.transform_factory = GcnTransformFactory()

        self.message_passer_factory = GcnMessagePasserFactory(self.feature_factory, self.transform_factory)

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

    def get_gated_gcn_nohypergraph(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_nohypergraph_message_passer
        hypergraph_gcn_propagation_units = self.make_gcn_nohypergraph_propagation(gcn_settings, hypergraph, message_passer_function, variables)
        return hypergraph_gcn_propagation_units

    def get_gated_gcn_with_relation_bag_features(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_message_passer_with_relation_bag_features
        hypergraph_gcn_propagation_units = self.make_gcn(gcn_settings, hypergraph, message_passer_function, variables)
        return hypergraph_gcn_propagation_units

    def get_gated_gcn_nohypergraph_with_relation_bag_features(self, hypergraph, variables, gcn_settings):
        message_passer_function = self.get_gated_nohypergraph_message_passer_with_relation_bag_features
        hypergraph_gcn_propagation_units = self.make_gcn_nohypergraph_propagation(gcn_settings, hypergraph, message_passer_function, variables)
        return hypergraph_gcn_propagation_units

    """
    Helpers:
    """

    def make_gcn(self, gcn_settings, hypergraph, message_passer_function, variables):
        gcn = GCN()

        weight_tying = False

        if weight_tying:
            pass
        else:
            for layer in range(gcn_settings["n_layers"]):
                propagation = HypergraphGcnPropagationUnit("layer_" + str(layer),
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
                self.add_message_passers(gcn_settings, propagation, hypergraph, message_passer_function, layer)
                self_connection = None
                gcn.add_layer(propagation, self_connection)

        return gcn

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

    def get_propagation_unit(self, gcn_setting, hypergraph, layer=None):
        prefix = "prop" if layer is None else "prop_"+str(layer)
        if gcn_setting["hypergraph_gcn"]:
            in_dimension, out_dimension = self.compute_dimensions(gcn_setting, layer)
            return NormalGcnPropagationUnit(prefix, in_dimension, out_dimension, hypergraph)
        else:
            HypergraphGcnPropagationUnit(prefix)

    def get_self_connection_unit(self, gcn_setting, hypergraph, layer=None):
        prefix = "self" if layer is None else "self_"+str(layer)
        in_dimension, out_dimension = self.compute_dimensions(gcn_setting, layer)
        if gcn_setting["self_connection_type"] == "highway":
            return HighwayUpdate(prefix, in_dimension, out_dimension, hypergraph)
        elif gcn_setting["self_connection_type"] == "cell":
            return CellUpdate(prefix, in_dimension, out_dimension, hypergraph)
        elif gcn_setting["self_connection_type"] == "residual":
            return ResidualUpdate(prefix, in_dimension, out_dimension, hypergraph)

    def compute_dimensions(self, gcn_setting, layer):
        out_dimension = gcn_setting["out_dimension"]
        if layer is None or layer == 0:
            in_dimension = gcn_setting["in_dimension"]
        else:
            in_dimension = gcn_setting["out_dimension"]

        return in_dimension, out_dimension

    def make_gcn_nohypergraph_propagation(self, gcn_settings, hypergraph, message_passer_function, variables):
        print(gcn_settings)
        gcn = GCN(gcn_settings)
        if gcn_settings["weight_tying"] > 1:
            pass
        else:
            for layer in range(gcn_settings["n_layers"]):
                propagation = self.get_propagation_unit(gcn_settings, hypergraph, layer=layer)
                self_connection = self.get_self_connection_unit(gcn_settings, hypergraph, layer=layer)
                self.add_message_passers(gcn_settings, propagation, hypergraph, message_passer_function, layer)

                gcn.add_layer(propagation, self_connection)

        return gcn

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

    def get_gated_type_only_message_passer(self, gcn_settings, hypergraph, message_instructions, current_layer):
        dimension = gcn_settings["embedding_dimension"]
        number_of_relation_types = gcn_settings["n_relation_types"]
        message_passer = GcnMessagePasser(hypergraph,
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

    def get_gated_message_passer(self, gcn_settings, hypergraph, message_instructions, current_layer):
        input_dim = self.get_hypergraph_dim(current_layer, gcn_settings, message_instructions)

        message_passer_settings = {"sender_tags": message_instructions["sender_tags"],
                                   "receiver_tags": message_instructions["receiver_tags"],
                                   "invert": message_instructions["invert"],
                                   "input_dimension": input_dim,
                                   "output_dimension": gcn_settings["embedding_dimension"],
                                   "use_relation_type_features": True,
                                   "relation_type_embedding_dimension": gcn_settings["relation_type_embedding_dimension"],
                                   "sentence_feature_dimension": gcn_settings["sentence_embedding_dimension"]}
        message_passer = self.message_passer_factory.get_message_passer(hypergraph, message_passer_settings)

        return message_passer

    def get_gated_nohypergraph_message_passer(self, gcn_settings, hypergraph, message_instructions, current_layer):
        input_dim = self.get_nohypergraph_dim(current_layer, gcn_settings)

        message_passer_settings = {"sender_tags": message_instructions["sender_tags"],
                                   "receiver_tags": message_instructions["receiver_tags"],
                                   "invert": message_instructions["invert"],
                                   "input_dimension": input_dim,
                                   "output_dimension": gcn_settings["embedding_dimension"],
                                   "use_relation_type_features": True,
                                   "relation_type_embedding_dimension": gcn_settings["relation_type_embedding_dimension"],
                                   "sentence_feature_dimension": gcn_settings["sentence_embedding_dimension"]}
        message_passer = self.message_passer_factory.get_message_passer(hypergraph, message_passer_settings)

        return message_passer

    def get_gated_message_passer_with_relation_bag_features(self, gcn_settings, hypergraph, message_instructions, current_layer):
        input_dim = self.get_hypergraph_dim(current_layer, gcn_settings, message_instructions)
        message_passer_settings = {"sender_tags": message_instructions["sender_tags"],
                                   "receiver_tags": message_instructions["receiver_tags"],
                                   "invert": message_instructions["invert"],
                                   "input_dimension": input_dim,
                                   "output_dimension": gcn_settings["embedding_dimension"],
                                   "use_relation_type_features": True,
                                   "relation_type_embedding_dimension": gcn_settings["relation_type_embedding_dimension"],
                                   "use_relation_part_features": True,
                                   "relation_part_embedding_dimension": gcn_settings["relation_part_embedding_dimension"],
                                   "sentence_feature_dimension": gcn_settings["sentence_embedding_dimension"]}
        message_passer = self.message_passer_factory.get_message_passer(hypergraph, message_passer_settings)

        return message_passer

    def get_gated_nohypergraph_message_passer_with_relation_bag_features(self, gcn_settings, hypergraph, message_instructions, current_layer):
        input_dim = self.get_nohypergraph_dim(current_layer, gcn_settings)
        message_passer_settings = {"sender_tags": message_instructions["sender_tags"],
                                   "receiver_tags": message_instructions["receiver_tags"],
                                   "invert": message_instructions["invert"],
                                   "input_dimension": input_dim,
                                   "output_dimension": gcn_settings["out_dimension"],
                                   "use_relation_type_features": True,
                                   "relation_type_embedding_dimension": gcn_settings["relation_type_embedding_dimension"],
                                   "use_relation_part_features": True,
                                   "relation_part_embedding_dimension": gcn_settings["relation_part_embedding_dimension"],
                                   "sentence_feature_dimension": gcn_settings["sentence_embedding_dimension"]}
        message_passer = self.message_passer_factory.get_message_passer(hypergraph, message_passer_settings)

        return message_passer


    def get_nohypergraph_dim(self, current_layer, gcn_settings):
        input_dim = gcn_settings["in_dimension"] if current_layer == 0 else gcn_settings["out_dimension"]
        return input_dim

    def get_hypergraph_dim(self, current_layer, gcn_settings, message_instructions):
        if message_instructions["sender_tags"] == "entities" and current_layer == 0:
            input_dim = 1
        else:
            input_dim = gcn_settings["embedding_dimension"]
        if message_instructions["receiver_tags"] == "entities" and current_layer == 0:
            input_dim += 1
        else:
            input_dim += gcn_settings["embedding_dimension"]
        return input_dim
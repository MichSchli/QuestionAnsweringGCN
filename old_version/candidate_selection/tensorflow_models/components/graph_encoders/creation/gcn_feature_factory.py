from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.external_batch_features import \
    ExternalBatchFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.receiver_features import \
    ReceiverFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_features.sender_features import SenderFeatures
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.edge_type_bow_transform import \
    TypeBowTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.type_bias_transform import \
    TypeBiasTransform


class GcnFeatureFactory:

    def get_features(self, hypergraph, feature_setting_list, gcn_instructions):
        return [self.build_feature(feature, hypergraph, gcn_instructions) for feature in feature_setting_list]

    def build_feature(self, feature_setting, hypergraph, gcn_instructions):
        if feature_setting["type"] == "sender_features":
            return SenderFeatures(hypergraph, gcn_instructions)

        if feature_setting["type"] == "receiver_features":
            return ReceiverFeatures(hypergraph, gcn_instructions)

        if feature_setting["type"] == "relation_type_embeddings":
            dimension = feature_setting["dimension"]
            n_relation_types = feature_setting["n_types"]
            return TypeBiasTransform(dimension, n_relation_types, hypergraph, gcn_instructions)

        if feature_setting["type"] == "relation_part_embeddings":
            dimension = feature_setting["dimension"]
            n_relation_parts = feature_setting["n_types"]
            return TypeBowTransform(dimension, n_relation_parts, hypergraph, gcn_instructions)

        if feature_setting["type"] == "external_sentence_features":
            return ExternalBatchFeatures(hypergraph, gcn_instructions, feature_setting["dimension"])

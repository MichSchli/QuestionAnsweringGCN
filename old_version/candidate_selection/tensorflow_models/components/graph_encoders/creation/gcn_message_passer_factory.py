from candidate_selection.tensorflow_models.components.graph_encoders.gcn_message_passer import GcnMessagePasser
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_gates import GcnGates
from candidate_selection.tensorflow_models.components.graph_encoders.subcomponents.gcn_messages import GcnMessages


class GcnMessagePasserFactory:

    def __init__(self, feature_factory, transform_factory):
        self.feature_factory = feature_factory
        self.transform_factory = transform_factory

    def get_message_passer(self, hypergraph, message_passer_settings):
        message_passer = GcnMessagePasser(hypergraph,
                                          senders=message_passer_settings["sender_tags"],
                                          receivers=message_passer_settings["receiver_tags"],
                                          gate_mode="type_key_comparison",
                                          inverse_edges=message_passer_settings["invert"])

        message_features = self.get_features(hypergraph, message_passer_settings)
        gate_features = self.get_features(hypergraph, message_passer_settings)

        external_sentence_feature = self.feature_factory.build_feature({"type": "external_sentence_features",
                                                                        "dimension": message_passer_settings["sentence_feature_dimension"]}, hypergraph, message_passer_settings)
        message_features.append(external_sentence_feature)
        gate_features.append(external_sentence_feature)
        message_passer.sentence_features = external_sentence_feature

        message_transforms = self.get_transforms(hypergraph, int(sum([feature.get_width() for feature in message_features])), message_passer_settings)
        gate_transforms = self.get_transforms(hypergraph, int(sum([feature.get_width() for feature in gate_features])), message_passer_settings, final_gate_transform=True)

        message_passer.messages = GcnMessages(message_features,
                                              message_transforms)
        message_passer.gates = GcnGates(gate_features,
                                        gate_transforms)

        return message_passer

    def get_features(self, hypergraph, message_passer_settings):
        feature_settings = [{"type": "sender_features"},
                            {"type": "receiver_features"}]

        #TODO: More-or-less random numbers for n types
        if message_passer_settings["use_relation_type_features"] == True:
            n_types = 2800 if message_passer_settings["sender_tags"] != "words" else 0
            feature_settings.append({"type": "relation_type_embeddings",
                                     "dimension": message_passer_settings["relation_type_embedding_dimension"],
                                     "n_types": n_types})

        if message_passer_settings["use_relation_part_features"] == True:
            feature_settings.append({"type": "relation_part_embeddings",
                                     "dimension": message_passer_settings["relation_part_embedding_dimension"],
                                     "n_types": 2800})

        return self.feature_factory.get_features(hypergraph, feature_settings, message_passer_settings)

    def get_transforms(self, hypergraph, feature_width, message_passer_settings, final_gate_transform=False):
        transform_settings = [{"type": "affine",
                               "input_dimension":feature_width,
                               "output_dimension": message_passer_settings["output_dimension"]},
                              {"type": "relu"}]

        if final_gate_transform:
            transform_settings.append({"type": "affine",
                                       "input_dimension": message_passer_settings["output_dimension"],
                                       "output_dimension": 1})

        return self.transform_factory.get_transforms(hypergraph, transform_settings, message_passer_settings)

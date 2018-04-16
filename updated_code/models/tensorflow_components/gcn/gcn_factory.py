from models.tensorflow_components.gcn.gcn import Gcn
from models.tensorflow_components.gcn.gcn_features.relation_features import RelationFeatures
from models.tensorflow_components.gcn.gcn_features.relation_part_features import RelationPartFeatures
from models.tensorflow_components.gcn.gcn_features.sentence_batch_features import SentenceBatchFeatures
from models.tensorflow_components.gcn.gcn_features.vertex_features import VertexFeatures
from models.tensorflow_components.gcn.gcn_gates.gcn_gates import GcnGates
from models.tensorflow_components.gcn.gcn_messages.gcn_messages import GcnMessages
from models.tensorflow_components.gcn.gcn_updaters.cell_state_updater import CellStateGcnUpdater
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent

from models.tensorflow_components.gcn.gcn_features.vertex_type_features import VertexTypeFeatures


class GcnFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get_dummy_gcn(self, graph, experiment_configuration):
        layers = int(experiment_configuration["gcn"]["layers"])
        relation_index = self.index_factory.get("relations", experiment_configuration)
        relation_index_width = relation_index.dimension

        relation_part_index = self.index_factory.get("relation_parts", experiment_configuration)
        relation_part_index_width = relation_part_index.dimension

        sentence_embedding_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        sentence_batch_features = SentenceBatchFeatures(graph, sentence_embedding_dim)

        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        sender_features = VertexFeatures(graph, "senders", gcn_dim)
        receiver_features = VertexFeatures(graph, "receivers", gcn_dim)
        sender_type_features = VertexTypeFeatures(graph, "sender")
        receiver_type_features = VertexTypeFeatures(graph, "receiver")
        relation_features = RelationFeatures(graph, relation_index_width, relation_index)
        relation_part_features = RelationPartFeatures(graph, relation_part_index_width, relation_part_index)


        message_features = [sender_features,
                            receiver_features,
                            sender_type_features,
                            receiver_type_features,
                            relation_features,
                            relation_part_features,
                            sentence_batch_features]
        gate_features = [sender_features,
                         receiver_features,
                         sender_type_features,
                         receiver_type_features,
                         relation_features,
                         relation_part_features,
                         sentence_batch_features]

        message_hidden_dims = [int(e) for e in experiment_configuration["gcn"]["message_hidden_dimension"].split("|")]
        gate_hidden_dims = [int(e) for e in experiment_configuration["gcn"]["gate_hidden_dimension"].split("|")]

        gcn_layers = [None]*layers
        for layer in range(layers):
            vertex_input_dim = gcn_dim if layer == 0 else gcn_dim
            message_feature_dimension = sum(m.get_width() for m in message_features)
            gate_feature_dimension = sum(g.get_width() for g in gate_features)

            message_dims = [message_feature_dimension] + message_hidden_dims + [gcn_dim]
            gate_dims = [gate_feature_dimension] + gate_hidden_dims + [1]

            message_perceptron = MultilayerPerceptronComponent(message_dims,
                                                               "message_mlp",
                                                               dropout_rate=float(experiment_configuration["regularization"]["gcn_dropout"]),
                                                               l2_scale=float(experiment_configuration["regularization"]["gcn_l2"]))
            gate_perceptron = MultilayerPerceptronComponent(gate_dims,
                                                            "gate_mlp",
                                                            dropout_rate=float(experiment_configuration["regularization"]["gcn_dropout"]),
                                                            l2_scale=float(experiment_configuration["regularization"]["gcn_l2"]))

            messages = GcnMessages(message_features,
                                   message_perceptron)
            gates = GcnGates(gate_features,
                             gate_perceptron,
                             l1_scale=float(experiment_configuration["regularization"]["gate_l1"]))

            updater = CellStateGcnUpdater("cell_state_"+str(layer), vertex_input_dim, gcn_dim, graph)

            gcn_layers[layer] = Gcn(messages, gates, updater, graph)
            sender_features.width = gcn_dim
            receiver_features.width = gcn_dim

        return gcn_layers, sentence_batch_features
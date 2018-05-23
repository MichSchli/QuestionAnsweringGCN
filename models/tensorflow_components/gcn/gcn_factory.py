from example_reader.graph_reader.edge_type_utils import EdgeTypeUtils
from models.tensorflow_components.gcn.gcn_features.gcn_type_feature_wrapper import GcnTypeFeatureWrapper
from models.tensorflow_components.gcn.gcn_gates.gcn_bias_and_feature_gates import GcnBiasAndFeatureGates
from models.tensorflow_components.gcn.gcn_gates.gcn_no_gates import GcnNoGates
from models.tensorflow_components.gcn.gcn_messages.gcn_bias_only_messages import GcnBiasOnlyMessages
from models.tensorflow_components.gcn.gcn_propagator import GcnPropagator
from models.tensorflow_components.gcn.gcn_features.max_score_features import VertexScoreFeatures
from models.tensorflow_components.gcn.gcn_features.relation_features import RelationFeatures
from models.tensorflow_components.gcn.gcn_features.relation_part_features import RelationPartFeatures
from models.tensorflow_components.gcn.gcn_features.sentence_batch_features import SentenceBatchFeatures
from models.tensorflow_components.gcn.gcn_features.vertex_features import VertexFeatures
from models.tensorflow_components.gcn.gcn_gates.gcn_gates import GcnGates
from models.tensorflow_components.gcn.gcn_messages.gcn_messages import GcnMessages
from models.tensorflow_components.gcn.gcn_updaters.cell_state_updater import CellStateGcnUpdater, \
    CellStateGcnInitializer
from models.tensorflow_components.gcn.gcn_updaters.copy_context_updater import CopyContextGcnInitializer, \
    CopyContextGcnUpdater
from models.tensorflow_components.gcn.gcn_updaters.simple_self_loop_updater import SimpleSelfLoopGcnInitializer, \
    SimpleSelfLoopGcnUpdater
from models.tensorflow_components.gcn.gcns.gcn import Gcn
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent

from models.tensorflow_components.gcn.gcn_features.vertex_type_features import VertexTypeFeatures


class GcnFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory
        self.edge_type_utils = EdgeTypeUtils()

    def get_message_bias_only_gcn(self, graph, experiment_configuration):
        layers = int(experiment_configuration["gcn"]["layers"])
        relation_index = self.index_factory.get("relations", experiment_configuration)
        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        initial_input_dim = gcn_dim + 6 + 1
        initial_cell_updater = CopyContextGcnInitializer("cell_state_initializer", initial_input_dim, gcn_dim, graph)

        gcn_layers = [None] * layers
        updaters = [None] * layers

        sentence_embedding_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        sentence_batch_features = SentenceBatchFeatures(graph, sentence_embedding_dim)

        add_backward_gcn = "inverse_relations" in experiment_configuration["architecture"] \
                           and experiment_configuration["architecture"]["inverse_relations"] == "separate"
        for layer in range(layers):
            message_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
            f_propagator = self.get_message_bias_only_propagator(message_bias, graph, direction="forward")
            gcn_layers[layer] = [f_propagator]

            if add_backward_gcn:
                b_message_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
                b_propagator = self.get_message_bias_only_propagator(b_message_bias, graph, direction="backward")
                gcn_layers[layer].append(b_propagator)

            updaters[layer] = CopyContextGcnUpdater("cell_state_" + str(layer), gcn_dim, gcn_dim, graph)

        gcn = Gcn(initial_cell_updater, gcn_layers, updaters)

        return gcn, sentence_batch_features

    def get_gated_message_bias_gcn(self, graph, experiment_configuration):
        layers = int(experiment_configuration["gcn"]["layers"])
        relation_index = self.index_factory.get("relations", experiment_configuration)
        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        sentence_embedding_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        sentence_batch_features = SentenceBatchFeatures(graph, sentence_embedding_dim)

        initial_input_dim = gcn_dim + 6 + 1
        initial_cell_updater = CellStateGcnInitializer("cell_state_initializer", initial_input_dim, gcn_dim, graph)

        gcn_layers = [None] * layers
        updaters = [None] * layers

        add_backward_gcn = "inverse_relations" in experiment_configuration["architecture"] \
                           and experiment_configuration["architecture"]["inverse_relations"] == "separate"
        for layer in range(layers):
            if experiment_configuration["architecture"]["separate_gcns"] == "All":
                gcn_layers[layer] = []
                for i in range(self.edge_type_utils.count_types()):
                    message_features = [VertexFeatures(graph, "senders", gcn_dim, edge_type=i),
                                        VertexFeatures(graph, "receivers", gcn_dim, edge_type=i)]
                    message_bias = [RelationFeatures(graph, gcn_dim, relation_index, edge_type=i)]

                    gate_features = [GcnTypeFeatureWrapper(sentence_batch_features, graph, i)]
                    gate_bias = [RelationFeatures(graph, gcn_dim, relation_index, edge_type=i)]
                    f_propagator = self.get_gated_message_bias_propagator(message_bias, gate_features, gate_bias, graph,
                                                                      experiment_configuration, str(layer),
                                                                      direction="forward", edge_type=i)
                    gcn_layers[layer].append(f_propagator)

                    if add_backward_gcn:
                        b_message_bias = [RelationFeatures(graph, gcn_dim, relation_index, edge_type=i)]
                        b_gate_bias = [RelationFeatures(graph, gcn_dim, relation_index, edge_type=i)]
                        b_propagator = self.get_gated_message_bias_propagator(b_message_bias, gate_features, b_gate_bias,
                                                                          graph, experiment_configuration, str(layer),
                                                                          direction="backward", edge_type=i)
                        gcn_layers[layer].append(b_propagator)

            else:
                gate_features = [sentence_batch_features]
                message_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
                gate_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
                f_propagator = self.get_gated_message_bias_propagator(message_bias, gate_features, gate_bias, graph, experiment_configuration, str(layer),direction="forward")
                gcn_layers[layer] = [f_propagator]

                if add_backward_gcn:
                    b_message_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
                    b_gate_bias = [RelationFeatures(graph, gcn_dim, relation_index)]
                    b_propagator = self.get_gated_message_bias_propagator(b_message_bias, gate_features, b_gate_bias, graph, experiment_configuration, str(layer),direction="backward")
                    gcn_layers[layer].append(b_propagator)

            updaters[layer] = CellStateGcnUpdater("cell_state_" + str(layer), gcn_dim, gcn_dim, graph)

        gcn = Gcn(initial_cell_updater, gcn_layers, updaters)

        return gcn, sentence_batch_features

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
        sender_score_features = VertexScoreFeatures(graph, "sender")
        receiver_score_features = VertexScoreFeatures(graph, "receiver")
        relation_features = RelationFeatures(graph, relation_index_width, relation_index)
        relation_part_features = RelationPartFeatures(graph, relation_part_index_width, relation_part_index)


        message_features = [sender_features,
                            receiver_features,
                            sender_type_features,
                            receiver_type_features,
                            relation_features,
                            relation_part_features] #,
                            #sentence_batch_features,
                            #sender_score_features,
                            #receiver_score_features]
        gate_features = [sender_features,
                         receiver_features,
                         sender_type_features,
                         receiver_type_features,
                         relation_features,
                         relation_part_features,
                         sentence_batch_features,
                         sender_score_features,
                         receiver_score_features]

        message_bias = [RelationFeatures(graph, gcn_dim, relation_index)]


        initial_input_dim = gcn_dim + 6 + 1
        initial_cell_updater = CellStateGcnInitializer("cell_state_initializer", initial_input_dim, gcn_dim, graph)

        gcn_layers = [None]*layers
        updaters = [None]*layers

        add_backward_gcn = "inverse_relations" in experiment_configuration["architecture"] \
                    and experiment_configuration["architecture"]["inverse_relations"] == "separate"
        for layer in range(layers):
            f_propagator = self.get_propagator(gate_features,
                                             message_features,
                                             graph,
                                             gcn_dim,
                                             experiment_configuration,
                                             str(layer),
                                             direction="forward")
            gcn_layers[layer] = [f_propagator]

            if add_backward_gcn:
                b_propagator = self.get_propagator(gate_features,
                                                   message_features,
                                                   graph,
                                                   gcn_dim,
                                                   experiment_configuration,
                                                   str(layer),
                                                   direction="backward")
                gcn_layers[layer].append(b_propagator)

            updaters[layer] = CellStateGcnUpdater("cell_state_"+str(layer), gcn_dim, gcn_dim, graph)

            sender_features.width = gcn_dim
            receiver_features.width = gcn_dim

        gcn = Gcn(initial_cell_updater, gcn_layers, updaters)

        return gcn, sentence_batch_features

    def get_propagator(self, gate_features, message_features, graph, gcn_dim, experiment_configuration, name, direction="forward", edge_type=None):
        message_feature_dimension = sum(m.get_width() for m in message_features)
        gate_feature_dimension = sum(g.get_width() for g in gate_features)
        message_hidden_dims = [int(e) for e in
                               experiment_configuration["gcn"]["message_hidden_dimension"].split("|")]
        gate_hidden_dims = [int(e) for e in experiment_configuration["gcn"]["gate_hidden_dimension"].split("|")]
        message_dims = [message_feature_dimension]  + [gcn_dim]#[message_feature_dimension] + message_hidden_dims + [gcn_dim]
        gate_dims = [gate_feature_dimension] + gate_hidden_dims + [1]

        message_perceptron = MultilayerPerceptronComponent(message_dims,
                                                           "message_mlp_"+direction+name,
                                                           dropout_rate=float(
                                                               experiment_configuration["regularization"][
                                                                   "gcn_dropout"]),
                                                           l2_scale=float(
                                                               experiment_configuration["regularization"]["gcn_l2"]))
        gate_perceptron = MultilayerPerceptronComponent(gate_dims,
                                                        "gate_mlp_"+direction+name,
                                                        dropout_rate=float(
                                                            experiment_configuration["regularization"]["gcn_dropout"]),
                                                        l2_scale=float(
                                                            experiment_configuration["regularization"]["gcn_l2"]))
        messages = GcnMessages(message_features,
                               message_perceptron)
        gates = GcnGates(gate_features,
                         gate_perceptron,
                         l1_scale=float(experiment_configuration["regularization"]["gate_l1"]))
        propagator = GcnPropagator(messages, gates, graph, direction)
        return propagator

    def get_message_bias_only_propagator(self, message_bias, graph, direction="forward"):
        messages = GcnBiasOnlyMessages(message_bias)
        gates = GcnNoGates()
        propagator = GcnPropagator(messages, gates, graph, direction)
        return propagator

    def get_gated_message_bias_propagator(self, message_bias, gate_features, gate_bias, graph, experiment_configuration, name, direction="forward", edge_type=None):
        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])
        gate_feature_dimension = sum(g.get_width() for g in gate_features)
        gate_dims_1 = [gate_feature_dimension, gcn_dim]
        gate_dims_2 = [gcn_dim, 1]
        gate_perceptron_1 = MultilayerPerceptronComponent(gate_dims_1,
                                                        "gate_mlp_"+direction+name,
                                                        dropout_rate=float(
                                                            experiment_configuration["regularization"]["gcn_dropout"]),
                                                        l2_scale=float(
                                                            experiment_configuration["regularization"]["gcn_l2"]))
        gate_perceptron_2 = MultilayerPerceptronComponent(gate_dims_2,
                                                        "gate_mlp_"+direction+name,
                                                        dropout_rate=float(
                                                            experiment_configuration["regularization"]["gcn_dropout"]),
                                                        l2_scale=float(
                                                            experiment_configuration["regularization"]["gcn_l2"]))
        messages = GcnBiasOnlyMessages(message_bias)
        gates = GcnBiasAndFeatureGates(gate_bias, gate_features, gate_perceptron_1, gate_perceptron_2, l1_scale=float(experiment_configuration["regularization"]["gate_l1"]))
        propagator = GcnPropagator(messages, gates, graph, direction, edge_type=edge_type)
        return propagator
from models.tensorflow_components.gcn.gcn import Gcn
from models.tensorflow_components.gcn.gcn_features.relation_features import RelationFeatures
from models.tensorflow_components.gcn.gcn_features.relation_part_features import RelationPartFeatures
from models.tensorflow_components.gcn.gcn_features.sentence_batch_features import SentenceBatchFeatures
from models.tensorflow_components.gcn.gcn_features.vertex_features import VertexFeatures
from models.tensorflow_components.gcn.gcn_gates.gcn_gates import GcnGates
from models.tensorflow_components.gcn.gcn_messages.gcn_messages import GcnMessages
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent


class GcnFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get_dummy_gcn(self, graph, experiment_configuration):
        relation_index = self.index_factory.get("relations", experiment_configuration)
        relation_index_width = relation_index.dimension

        relation_part_index = self.index_factory.get("relation_parts", experiment_configuration)
        relation_part_index_width = relation_part_index.dimension

        sentence_batch_features = SentenceBatchFeatures(graph, 100)

        message_features = [VertexFeatures(graph, "senders", 1),
                            VertexFeatures(graph, "receivers", 1),
                            RelationFeatures(graph, relation_index_width, relation_index),
                            RelationPartFeatures(graph, relation_part_index_width, relation_part_index),
                            sentence_batch_features]
        gate_features = [VertexFeatures(graph, "senders", 1),
                         VertexFeatures(graph, "receivers", 1),
                         RelationFeatures(graph, relation_index_width, relation_index),
                         RelationPartFeatures(graph, relation_part_index_width, relation_part_index),
                         sentence_batch_features]

        messages = GcnMessages(message_features, MultilayerPerceptronComponent([222,100], "mlp"))
        gates = GcnGates(gate_features, MultilayerPerceptronComponent([222, 200, 1], "mlp"))
        gcn = Gcn(messages, gates, None, graph)

        return gcn, sentence_batch_features
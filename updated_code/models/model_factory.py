from models.dummy_model import DummyModel
from models.dummy_tensorflow_model import DummyTensorflowModel
from models.tensorflow_components.aux_components.sentence_to_entity_mapper import SentenceToEntityMapper
from models.tensorflow_components.gcn.gcn_factory import GcnFactory
from models.tensorflow_components.graph.graph_component import GraphComponent
from models.tensorflow_components.loss_functions.sigmoid_loss import SigmoidLoss
from models.tensorflow_components.sentence.sentence_batch_component import SentenceBatchComponent
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent


class ModelFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory
        self.gcn_factory = GcnFactory(index_factory)

    def get(self, experiment_configuration):
        word_index = self.index_factory.get("words", experiment_configuration)
        pos_index = self.index_factory.get("pos", experiment_configuration)

        model = DummyTensorflowModel()
        model.graph = GraphComponent()
        model.add_component(model.graph)

        model.sentence = SentenceBatchComponent(word_index, pos_index)
        model.add_component(model.sentence)

        model.mlp = MultilayerPerceptronComponent([210,5,1], "mlp")
        model.add_component(model.mlp)

        model.sentence_to_entity_mapper = SentenceToEntityMapper()
        model.add_component(model.sentence_to_entity_mapper)

        model.loss = SigmoidLoss()
        model.add_component(model.loss)

        learning_rate = float(experiment_configuration["training"]["learning_rate"])
        gradient_clipping = float(experiment_configuration["training"]["gradient_clipping"])

        model.learning_rate = learning_rate
        model.gradient_clipping = gradient_clipping

        model.gcn, model.sentence_batch_features = self.gcn_factory.get_dummy_gcn(model.graph, experiment_configuration)
        model.add_component(model.gcn)
        model.add_component(model.sentence_batch_features)

        return model
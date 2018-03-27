from models.dummy_model import DummyModel
from models.dummy_tensorflow_model import DummyTensorflowModel
from models.tensorflow_components.graph.graph_component import GraphComponent
from models.tensorflow_components.loss_functions.sigmoid_loss import SigmoidLoss
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent


class ModelFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        model = DummyTensorflowModel()
        model.graph = GraphComponent()
        model.add_component(model.graph)

        model.mlp = MultilayerPerceptronComponent([2,5,1], "mlp")
        model.add_component(model.mlp)

        model.loss = SigmoidLoss()
        model.add_component(model.loss)

        return model
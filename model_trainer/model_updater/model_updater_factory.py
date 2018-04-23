from model_trainer.model_updater.dummy_model_updater import DummyModelUpdater
from model_trainer.model_updater.tensorflow_model_updater import TensorflowModelUpdater


class ModelUpdaterFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        learning_rate = float(experiment_configuration["training"]["learning_rate"])
        gradient_clipping = float(experiment_configuration["training"]["gradient_clipping"])
        return TensorflowModelUpdater(learning_rate, gradient_clipping)
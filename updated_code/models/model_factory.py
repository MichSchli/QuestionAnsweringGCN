from models.dummy_model import DummyModel


class ModelFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        return DummyModel()
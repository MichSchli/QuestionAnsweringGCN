from model_trainer.model_updater.dummy_model_updater import DummyModelUpdater


class ModelUpdaterFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        return DummyModelUpdater()
from model_trainer.model_trainer import ModelTrainer
from model_trainer.model_updater.model_updater_factory import ModelUpdaterFactory


class ModelTrainerFactory:

    example_reader_factory = None
    example_extender_factory = None
    example_batcher_factory = None
    model_updater = None

    def __init__(self, example_reader_factory, example_extender_factory, example_batcher_factory):
        self.example_reader_factory = example_reader_factory
        self.example_extender_factory = example_extender_factory
        self.example_batcher_factory = example_batcher_factory
        self.model_updater_factory = ModelUpdaterFactory()

    def get(self, experiment_configuration):
        example_reader = self.example_reader_factory.get(experiment_configuration)
        example_extender = self.example_extender_factory.get(experiment_configuration, "train")
        example_batcher = self.example_batcher_factory.get(experiment_configuration, "train")
        model_updater = self.model_updater_factory.get(experiment_configuration)

        model_trainer = ModelTrainer(example_reader, example_extender, example_batcher, model_updater)
        model_trainer.max_iterations = int(experiment_configuration["training"]["max_iterations"])

        return model_trainer
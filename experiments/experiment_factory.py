from example_batcher.example_batcher_factory import ExampleBatcherFactory
from example_extender.example_extender_factory import ExampleExtenderFactory
from example_reader.example_reader_factory import ExampleReaderFactory
from experiments.experiment import Experiment
from indexing.index_factory import IndexFactory
from model_tester.model_tester_factory import ModelTesterFactory
from model_trainer.model_trainer_factory import ModelTrainerFactory
from models.model_factory import ModelFactory


class ExperimentFactory:

    example_reader_factory = None
    example_extender_factory = None
    index_factory = None
    model_factory = None
    model_updater_factory = None

    def __init__(self, logger):
        self.logger = logger
        self.index_factory = IndexFactory()
        self.example_reader_factory = ExampleReaderFactory(self.index_factory)
        self.example_extender_factory = ExampleExtenderFactory(self.index_factory)
        self.example_batcher_factory = ExampleBatcherFactory()
        self.model_factory = ModelFactory(self.index_factory)
        self.model_tester_factory = ModelTesterFactory(self.example_reader_factory, self.example_extender_factory, self.example_batcher_factory, self.index_factory)
        self.model_trainer_factory = ModelTrainerFactory(self.example_reader_factory,
                                                         self.example_extender_factory,
                                                         self.example_batcher_factory,
                                                         self.model_tester_factory,
                                                         self.logger)

    def get(self, experiment_configuration):
        model_trainer = self.model_trainer_factory.get(experiment_configuration)
        model_tester = self.model_tester_factory.get(experiment_configuration)

        model = self.model_factory.get(experiment_configuration)

        experiment = Experiment(model_trainer, model_tester, model, self.logger)

        return experiment
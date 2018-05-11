from model_tester.evaluator.evaluator_factory import EvaluatorFactory
from model_tester.model_tester import ModelTester
from model_trainer.model_trainer import ModelTrainer


class ModelTesterFactory:

    example_reader_factory = None
    example_extender_factory = None
    example_batcher_factory = None
    evaluator = None

    def __init__(self, example_reader_factory, example_extender_factory, example_batcher_factory, index_factory):
        self.example_reader_factory = example_reader_factory
        self.example_extender_factory = example_extender_factory
        self.example_batcher_factory = example_batcher_factory
        self.evaluator_factory = EvaluatorFactory(index_factory)

    def get(self, experiment_configuration):
        example_reader = self.example_reader_factory.get(experiment_configuration)
        example_extender = self.example_extender_factory.get(experiment_configuration, "test")
        example_batcher = self.example_batcher_factory.get(experiment_configuration, "test")
        evaluator = self.evaluator_factory.get(experiment_configuration)

        model_tester = ModelTester(example_reader, example_extender, example_batcher, evaluator)

        return model_tester
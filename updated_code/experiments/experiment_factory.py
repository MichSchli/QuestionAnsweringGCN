from example_extender.example_extender_factory import ExampleExtenderFactory
from example_reader.example_reader_factory import ExampleReaderFactory
from experiments.experiment import Experiment
from indexing.index_factory import IndexFactory


class ExperimentFactory:

    example_reader_factory = None
    example_extender_factory = None
    index_factory = None

    def __init__(self):
        self.index_factory = IndexFactory()
        self.example_reader_factory = ExampleReaderFactory(self.index_factory)
        self.example_extender_factory = ExampleExtenderFactory()

    def get(self, experiment_configuration):
        example_reader = self.example_reader_factory.get(experiment_configuration)
        example_extender = self.example_extender_factory.get(experiment_configuration)
        experiment = Experiment(example_reader, example_extender)

        return experiment
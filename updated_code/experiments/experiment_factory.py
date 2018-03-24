from example_reader.example_reader_factory import ExampleReaderFactory
from experiments.experiment import Experiment
from indexing.index_factory import IndexFactory


class ExperimentFactory:

    example_reader_factory = None
    index_factory = None

    def __init__(self):
        self.index_factory = IndexFactory()
        self.example_reader_factory = ExampleReaderFactory(self.index_factory)

    def get(self, experiment_configuration):
        example_reader = self.example_reader_factory.get(experiment_configuration)
        experiment = Experiment(example_reader)

        return experiment
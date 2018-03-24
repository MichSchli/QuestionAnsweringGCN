from example_reader.example_reader import ExampleReader
from example_reader.graph_reader.graph_reader_factory import GraphReaderFactory
from example_reader.question_reader.question_reader_factory import QuestionReaderFactory


class ExampleReaderFactory:

    question_reader_factory = None
    graph_reader_factory = None

    def __init__(self, index_factory):
        self.question_reader_factory = QuestionReaderFactory()
        self.graph_reader_factory = GraphReaderFactory(index_factory)

    def get(self, experiment_configuration):
        question_reader = self.question_reader_factory.get(experiment_configuration)
        graph_reader = self.graph_reader_factory.get(experiment_configuration)
        return ExampleReader(question_reader, graph_reader)
from example_reader.example_reader import ExampleReader
from example_reader.gold_answer_reader.gold_answer_reader_factory import GoldAnswerReaderFactory
from example_reader.gold_path_reader.gold_path_reader_factory import GoldPathReaderFactory
from example_reader.graph_reader.graph_reader_factory import GraphReaderFactory
from example_reader.mention_reader.mention_reader_factory import MentionReaderFactory
from example_reader.question_reader.question_reader_factory import QuestionReaderFactory


class ExampleReaderFactory:

    question_reader_factory = None
    graph_reader_factory = None
    mention_reader_factory = None

    def __init__(self, index_factory):
        self.question_reader_factory = QuestionReaderFactory(index_factory)
        self.graph_reader_factory = GraphReaderFactory(index_factory)
        self.mention_reader_factory = MentionReaderFactory()
        self.gold_answer_reader_factory = GoldAnswerReaderFactory()
        self.gold_path_reader_factory = GoldPathReaderFactory()

    def get(self, experiment_configuration):
        dataset_map = {'train': experiment_configuration['dataset']['train_file'],
                       'valid': experiment_configuration['dataset']['valid_file'],
                       'test': experiment_configuration['dataset']['test_file']}

        question_reader = self.question_reader_factory.get(experiment_configuration)
        mention_reader = self.mention_reader_factory.get(experiment_configuration)
        graph_reader = self.graph_reader_factory.get(experiment_configuration)
        gold_answer_reader = self.gold_answer_reader_factory.get(experiment_configuration)
        gold_path_reader = self.gold_path_reader_factory.get(experiment_configuration)

        example_reader = ExampleReader(question_reader, graph_reader, mention_reader, gold_answer_reader, gold_path_reader, dataset_map)
        example_reader.project_names = experiment_configuration["endpoint"]["project_names"] == "True"
        return example_reader


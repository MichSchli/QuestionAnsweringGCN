from example_reader.question_reader.question_indexer import QuestionIndexer
from example_reader.question_reader.question_reader import QuestionReader


class QuestionReaderFactory:
    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration):
        word_index = self.index_factory.get("words", experiment_configuration)
        pos_index = self.index_factory.get("pos", experiment_configuration)
        question_reader = QuestionReader()
        question_indexer = QuestionIndexer(question_reader, word_index, pos_index)
        return question_indexer
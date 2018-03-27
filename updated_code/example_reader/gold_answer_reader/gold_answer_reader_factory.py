from example_reader.gold_answer_reader.gold_answer_reader import GoldAnswerReader


class GoldAnswerReaderFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        return GoldAnswerReader()
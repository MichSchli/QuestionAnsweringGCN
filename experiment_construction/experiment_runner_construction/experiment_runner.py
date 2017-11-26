from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader


class ExperimentRunner:

    learner = None
    train_file_iterator = None
    kb_prefix = None

    def __init__(self):
        self.kb_prefix = ""

    def set_train_file(self, train_file_location):
        self.train_file_iterator = ConllReader(train_file_location, self.kb_prefix)

    def set_kb_prefix(self, prefix):
        self.kb_prefix = prefix

    def train_and_validate(self):
        self.learner.initialize()
        best_epochs, performance = self.learner.train_and_validate(self.train_file_iterator)
        return best_epochs, performance

    def evaluate(self, file):
        iterator = ConllReader(file, self.kb_prefix)

        predictions = self.learner.predict(iterator)
        evaluator = Evaluator(ConllReader(file), self.kb_prefix)

        evaluation = evaluator.evaluate(predictions)
        return evaluation.micro_f1
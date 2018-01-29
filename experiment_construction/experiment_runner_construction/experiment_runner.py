from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader


class ExperimentRunner:

    learner = None
    train_file_iterator = None
    kb_prefix = None
    max_elements_to_read = None

    def __init__(self, disambiguation=None, score_transform=None):
        self.kb_prefix = ""
        self.disambiguation = disambiguation
        self.score_transform = score_transform

    def set_train_file(self, train_file_location):
        self.train_file_iterator = ConllReader(train_file_location, self.kb_prefix,
                                               max_elements=self.max_elements_to_read,
                                               disambiguation=self.disambiguation,
                                               score_transform=self.score_transform)

    def set_validation_file(self, validation_file_location):
        self.validation_file_iterator = ConllReader(validation_file_location, self.kb_prefix,
                                                    max_elements=self.max_elements_to_read,
                                                    disambiguation=self.disambiguation,
                                                    score_transform=self.score_transform)

    def set_test_file(self, test_file_location):
        self.test_file_iterator = ConllReader(test_file_location, self.kb_prefix,
                                              max_elements=self.max_elements_to_read,
                                              disambiguation=self.disambiguation,
                                              score_transform=self.score_transform)

    def set_train_evaluator(self, train_evaluator):
        self.train_evaluator = train_evaluator

    def set_test_evaluator(self, test_evaluator):
        self.test_evaluator = test_evaluator

    def set_valid_evaluator(self, valid_evaluator):
        self.valid_evaluator = valid_evaluator

    def set_kb_prefix(self, prefix):
        self.kb_prefix = prefix

    def limit_elements(self, limit):
        self.max_elements_to_read = limit

    def train_and_validate(self):
        self.learner.initialize()
        best_epochs, performance = self.learner.train_and_validate(self.train_file_iterator, self.validation_file_iterator)
        return best_epochs, performance

    def evaluate(self, file):
        if file == "train_file":
            iterator = self.train_file_iterator
            evaluator = self.train_evaluator
        elif file == "valid_file":
            iterator = self.validation_file_iterator
            evaluator = self.valid_evaluator
        elif file == "test_file":
            iterator = self.test_file_iterator
            evaluator = self.test_evaluator

        predictions = self.learner.predict(iterator)
        evaluation = evaluator.evaluate(predictions)
        return evaluation
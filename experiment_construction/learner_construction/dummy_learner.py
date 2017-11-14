from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader


class DummyLearner:

    preprocessor = None
    candidate_generator = None
    model = None


    def __init__(self, validation_file_location):
        self.validation_file_iterator = ConllReader(validation_file_location)
        self.evaluator = Evaluator(self.validation_file_iterator)

    def update_setting(self, setting_string, value):
        pass

    def initialize(self):
        pass

    def predict(self, test_file_iterator):
        epoch_iterator = test_file_iterator.iterate()
        epoch_iterator = self.candidate_generator.enrich(epoch_iterator)

        for element in epoch_iterator:
            result = self.model.predict(element)
            yield result

    def train_and_validate(self, train_file):
        prediction = self.predict(self.validation_file_iterator)
        evaluation = self.evaluator.evaluate(prediction)
        return evaluation.micro_f1


    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_candidate_generator(self, candidate_generator):
        self.candidate_generator = candidate_generator

    def set_candidate_selector(self, candidate_selector):
        self.model = candidate_selector
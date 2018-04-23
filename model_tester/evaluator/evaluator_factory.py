from model_tester.evaluator.evaluator import Evaluator


class EvaluatorFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration):
        return Evaluator()
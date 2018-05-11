from model_tester.evaluator.evaluator import Evaluator


class EvaluatorFactory:

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration):
        relation_index = self.index_factory.get("relations", experiment_configuration)
        return Evaluator(relation_index)
from model_tester.evaluator.entity_prediction_evaluator.evaluator import Evaluator
from model_tester.evaluator.relation_prediction_evaluator.relation_prediction_evaluator import \
    RelationPredictionEvaluator
from model_tester.evaluator.relation_prediction_evaluator.relation_prediction_freebase_entity_evaluator import \
    RelationPredictionFreebaseEntityEvaluator


class EvaluatorFactory:

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration):

        if experiment_configuration["testing"]["evaluation"] == "entity_prediction":
            relation_index = self.index_factory.get("relations", experiment_configuration)
            return Evaluator(relation_index)
        elif experiment_configuration["testing"]["evaluation"] == "relation_prediction":
            return RelationPredictionEvaluator()
        elif experiment_configuration["testing"]["evaluation"] == "relation_prediction_to_entity":
            return RelationPredictionFreebaseEntityEvaluator(self.index_factory.get("relations", experiment_configuration))
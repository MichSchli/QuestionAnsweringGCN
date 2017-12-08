from experiment_construction.evaluator_construction.evaluator import Evaluator
from helpers.read_conll_files import ConllReader


class EvaluatorFactory:

    logger = None

    def __init__(self, logger):
        self.logger = logger

    def construct_evaluator(self, settings, file):
        evaluation_type = settings["evaluation"]["type"]
        method = settings["evaluation"]["method"]
        file_reader = ConllReader(settings["dataset"][file], entity_prefix=settings["endpoint"]["prefix"])

        if evaluation_type == "cutoff":
            cutoff = float(settings["evaluation"]["cutoff"])
            evaluator = Evaluator(file_reader, cutoff, self.logger, method=method)

        return evaluator
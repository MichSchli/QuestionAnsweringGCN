import copy

from experiment_construction.candidate_generator_construction.candidate_generator_factory import \
    CandidateGeneratorFactory
from experiment_construction.candidate_selector_construction.candidate_selector_factory import CandidateSelectorFactory
from experiment_construction.evaluator_construction.evaluator_factory import EvaluatorFactory
from experiment_construction.example_processor_construction.example_processor_factory import ExampleProcessorFactory
from experiment_construction.experiment_runner_construction.experiment_runner_factory import ExperimentRunnerFactory
from experiment_construction.fact_construction.fact_factory import FactFactory
from experiment_construction.index_construction.index_holder_factory import IndexHolderFactory
from experiment_construction.learner_construction.learner_factory import LearnerFactory
from experiment_construction.preprocessor_construction.preprocessor_factory import PreprocessorFactory
from experiment_construction.search.greedy import GreedySearch
from experiment_construction.search.grid import GridSearch
from helpers.static import Static


class ExperimentFactory:

    settings = None
    index_factory = None
    latest_experiment_runner = None

    def __init__(self, settings):
        self.settings = settings
        self.index_factory = IndexHolderFactory()
        self.preprocessor_factory = PreprocessorFactory(self.index_factory)
        self.candidate_generator_factory = CandidateGeneratorFactory(self.index_factory)
        self.fact_factory = FactFactory()
        self.candidate_selector_factory = CandidateSelectorFactory(self.index_factory, self.fact_factory)
        self.example_processor_factory = ExampleProcessorFactory()
        self.evaluator_factory = EvaluatorFactory(Static.logger)
        self.learner_factory = LearnerFactory(self.evaluator_factory,
                                              self.preprocessor_factory,
                                              self.candidate_generator_factory,
                                              self.candidate_selector_factory,
                                              self.example_processor_factory)
        self.experiment_runner_factory = ExperimentRunnerFactory(self.evaluator_factory, self.learner_factory)

    def search(self):
        strategy = self.iterate_settings()
        next_configuration = strategy.next(None)

        best_performance = -1
        best_string = None
        best_configuration = None

        while next_configuration is not None:
            epochs, parameter_string, performance = self.train_and_validate(next_configuration)

            if performance > best_performance:
                best_performance = performance
                best_string = parameter_string
                best_configuration = copy.deepcopy(next_configuration)
                best_configuration["training"]["max_epochs"] = str(epochs)
                best_configuration["training"]["use_early_stopping"] = "False"

            previous_performance = performance
            next_configuration = strategy.next(previous_performance)

        Static.logger.write("Parameter tuning done.", "experiment", "messages")
        Static.logger.write("Best setting: " + best_string, "experiment", "messages")
        Static.logger.write("Performance: " + str(best_performance), "experiment", "messages")

        return best_configuration

    def train_and_validate(self, next_configuration, report_running_train_performance=True):
        config_items = []
        for h, cfg in next_configuration.items():
            for k, v in cfg.items():
                config_items.append(h + ":" + k + "=" + v)
        parameter_string = ", ".join(config_items)
        Static.logger.write(parameter_string, "experiment", "parameters")
        experiment_runner = self.experiment_runner_factory.construct_experiment_runner(next_configuration)
        best_epochs, performance = experiment_runner.train_and_validate()
        self.latest_experiment_runner = experiment_runner

        if report_running_train_performance:
            self.evaluate("train_file")

        return best_epochs, parameter_string, performance

    def evaluate(self, dataset_string):
        Static.logger.write("Evaluating on \""+dataset_string+"\"...", "experiment", "messages")
        evaluation = self.latest_experiment_runner.evaluate(dataset_string)
        Static.logger.write(evaluation.summarize(), "experiment", "messages")

    """
    Iterate all possible configurations of settings:
    """
    def iterate_settings(self):
        use_grid_search = False
        if "training" in self.settings:
            if "search" in self.settings["training"]:
                use_grid_search = True if self.settings["training"]["search"] == "grid" else False

        if use_grid_search:
            return GridSearch(self.settings)
        else:
            return GreedySearch(self.settings)


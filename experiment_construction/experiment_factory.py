from experiment_construction.candidate_generator_construction.candidate_generator_factory import \
    CandidateGeneratorFactory
from experiment_construction.candidate_selector_construction.candidate_selector_factory import CandidateSelectorFactory
from experiment_construction.experiment_runner_construction.experiment_runner_factory import ExperimentRunnerFactory
from experiment_construction.index_construction.index_factory import IndexFactory
from experiment_construction.learner_construction.learner_factory import LearnerFactory
from experiment_construction.preprocessor_construction.preprocessor_factory import PreprocessorFactory
from experiment_construction.search.greedy import GreedySearch
from experiment_construction.search.grid import GridSearch


class ExperimentFactory:

    settings = None
    index_factory = None

    def __init__(self, settings):
        self.settings = settings
        self.index_factory = IndexFactory()
        self.preprocessor_factory = PreprocessorFactory()
        self.candidate_generator_factory = CandidateGeneratorFactory()
        self.candidate_selector_factory = CandidateSelectorFactory()
        self.learner_factory = LearnerFactory()
        self.experiment_runner_factory = ExperimentRunnerFactory()

    def search(self):
        strategy = self.iterate_settings()
        next_configuration = strategy.next(None)

        while next_configuration is not None:

            config_items = []
            for h,cfg in next_configuration.items():
                for k,v in cfg.items():
                    config_items.append(h+":"+k+"="+v)
            parameter_string = ",".join(config_items)
            print(parameter_string)

            indexers = self.index_factory.construct_indexes(next_configuration)
            preprocessors = self.preprocessor_factory.construct_preprocessor(indexers, next_configuration)
            candidate_generator = self.candidate_generator_factory.construct_candidate_generator(indexers, next_configuration)
            candidate_selector = self.candidate_selector_factory.construct_candidate_selector(indexers, preprocessors, next_configuration)
            learner = self.learner_factory.construct_learner(preprocessors, candidate_generator, candidate_selector, next_configuration)
            experiment_runner = self.experiment_runner_factory.construct_experiment_runner(preprocessors, learner, next_configuration)

            performance = experiment_runner.train_and_validate()

            print(performance)

            previous_performance = performance
            next_configuration = strategy.next(previous_performance)


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


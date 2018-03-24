"""
Tensorflow vs. e.g. oracle, train or no, that kind of stuff

May want to refactor name
"""
from experiment_construction.learner_construction.dummy_learner import DummyLearner
from experiment_construction.learner_construction.tensorflow_learner import TensorflowModel
from experiment_construction.learner_construction.validation_set_evaluator import ValidationSetEvaluator


class LearnerFactory:

    def __init__(self, evaluator_factory, preprocessor_factory, candidate_generator_factory, candidate_selector_factory, example_processor_factory, index_factory):
        self.evaluator_factory = evaluator_factory
        self.preprocessor_factory = preprocessor_factory
        self.candidate_generator_factory = candidate_generator_factory
        self.candidate_selector_factory = candidate_selector_factory
        self.example_processor_factory = example_processor_factory
        self.index_factory = index_factory

    def construct_learner(self, settings):
        preprocessor = self.preprocessor_factory.construct_preprocessor(settings)
        candidate_generator = self.candidate_generator_factory.construct_candidate_generator(settings)
        candidate_selector = self.candidate_selector_factory.construct_candidate_selector(settings)
        example_processor = self.example_processor_factory.construct_example_processor(settings)
        index = self.index_factory.construct_indexes(settings)

        learner = self.get_base_learner(candidate_selector, settings)
        learner.set_preprocessor(preprocessor)
        learner.set_candidate_generator(candidate_generator)
        learner.set_candidate_selector(candidate_selector)
        learner.set_example_processor(example_processor)
        learner.set_relation_indexer(index.relation_indexer)

        if "early_stopping" in settings["training"] or "epochs_between_tests" in settings["training"]:
            evaluator = self.evaluator_factory.construct_evaluator(settings, "valid_file")
            learner = ValidationSetEvaluator(learner, evaluator)

        for k, v in settings["training"].items():
            learner.update_setting(k, v)

        return learner

    def get_base_learner(self, candidate_selector, settings):
        if candidate_selector.is_tensorflow:
            return TensorflowModel()
        else:
            return DummyLearner(settings["dataset"]["valid_file"], settings["endpoint"]["prefix"])
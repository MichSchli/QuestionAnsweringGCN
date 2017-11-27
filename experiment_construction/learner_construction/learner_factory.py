"""
Tensorflow vs. e.g. oracle, train or no, that kind of stuff

May want to refactor name
"""
from experiment_construction.learner_construction.dummy_learner import DummyLearner
from experiment_construction.learner_construction.tensorflow_learner import TensorflowModel
from experiment_construction.learner_construction.validation_set_evaluator import ValidationSetEvaluator


class LearnerFactory:

    def construct_learner(self, preprocessor, candidate_generator, candidate_selector, settings):
        learner = self.get_base_learner(candidate_selector, settings)
        learner.set_preprocessor(preprocessor)
        learner.set_candidate_generator(candidate_generator)
        learner.set_candidate_selector(candidate_selector)

        if "early_stopping" in settings["training"] or "epochs_between_tests" in settings["training"]:
            if "prefix" in settings["endpoint"]:
                prefix = settings["endpoint"]["prefix"]
            else:
                prefix = ""

            learner = ValidationSetEvaluator(learner, settings["dataset"]["valid_file"], kb_prefix=prefix)

        for k, v in settings["training"].items():
            learner.update_setting(k, v)

        return learner

    def get_base_learner(self, candidate_selector, settings):
        if candidate_selector.is_tensorflow:
            return TensorflowModel()
        else:
            return DummyLearner(settings["dataset"]["valid_file"], settings["endpoint"]["prefix"])
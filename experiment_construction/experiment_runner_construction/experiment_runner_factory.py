from experiment_construction.experiment_runner_construction.experiment_runner import ExperimentRunner


class ExperimentRunnerFactory:

    def __init__(self, evaluator_factory, learner_factory):
        self.evaluator_factory = evaluator_factory
        self.learner_factory = learner_factory

    def construct_experiment_runner(self, settings):
        learner = self.learner_factory.construct_learner(settings)
        disambiguation = settings["other"]["disambiguation"]
        score_transform = settings["other"]["transform_disambiguation_scores"]
        experiment_runner = ExperimentRunner(disambiguation=disambiguation, score_transform=score_transform)
        experiment_runner.learner = learner

        if "prefix" in settings["endpoint"]:
            experiment_runner.set_kb_prefix(settings["endpoint"]["prefix"])

        experiment_runner.set_train_file(settings["dataset"]["train_file"])
        experiment_runner.set_validation_file(settings["dataset"]["valid_file"])
        experiment_runner.set_test_file(settings["dataset"]["test_file"])

        experiment_runner.set_train_evaluator(self.evaluator_factory.construct_evaluator(settings, "train_file"))
        experiment_runner.set_valid_evaluator(self.evaluator_factory.construct_evaluator(settings, "valid_file"))
        experiment_runner.set_test_evaluator(self.evaluator_factory.construct_evaluator(settings, "test_file"))

        return experiment_runner
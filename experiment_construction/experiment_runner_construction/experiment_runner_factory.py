from experiment_construction.experiment_runner_construction.experiment_runner import ExperimentRunner


class ExperimentRunnerFactory:

    def __init__(self, evaluator_factory):
        self.evaluator_factory = evaluator_factory

    def construct_experiment_runner(self, preprocessors, learner, settings):
        experiment_runner = ExperimentRunner()
        experiment_runner.learner = learner
        experiment_runner.limit_elements(3)

        if "prefix" in settings["endpoint"]:
            experiment_runner.set_kb_prefix(settings["endpoint"]["prefix"])

        experiment_runner.set_train_file(settings["dataset"]["train_file"])
        experiment_runner.set_validation_file(settings["dataset"]["valid_file"])
        experiment_runner.set_test_file(settings["dataset"]["test_file"])

        experiment_runner.set_train_evaluator(self.evaluator_factory.construct_evaluator(settings, "train_file"))
        experiment_runner.set_valid_evaluator(self.evaluator_factory.construct_evaluator(settings, "valid_file"))
        experiment_runner.set_test_evaluator(self.evaluator_factory.construct_evaluator(settings, "test_file"))

        return experiment_runner
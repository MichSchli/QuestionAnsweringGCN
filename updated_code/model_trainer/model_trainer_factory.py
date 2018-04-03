from model_trainer.model_trainer import ModelTrainer


class ModelTrainerFactory:

    example_reader_factory = None
    example_extender_factory = None
    example_batcher_factory = None
    model_tester_factory = None
    logger = None

    def __init__(self, example_reader_factory, example_extender_factory, example_batcher_factory, model_tester_factory, logger):
        self.example_reader_factory = example_reader_factory
        self.example_extender_factory = example_extender_factory
        self.example_batcher_factory = example_batcher_factory
        self.model_tester_factory = model_tester_factory
        self.logger = logger

    def get(self, experiment_configuration):
        example_reader = self.example_reader_factory.get(experiment_configuration)
        example_extender = self.example_extender_factory.get(experiment_configuration, "train")
        example_batcher = self.example_batcher_factory.get(experiment_configuration, "train")

        validation_evaluator = self.model_tester_factory.get(experiment_configuration)

        model_trainer = ModelTrainer(example_reader, example_extender, example_batcher, validation_evaluator, self.logger)
        model_trainer.max_iterations = int(experiment_configuration["training"]["max_iterations"])
        model_trainer.validate_every_n = int(experiment_configuration["training"]["validate_every_n"])
        model_trainer.report_loss_every_n = int(experiment_configuration["training"]["report_loss_every_n"])
        model_trainer.early_stopping = experiment_configuration["training"]["early_stopping"] == "True"

        return model_trainer
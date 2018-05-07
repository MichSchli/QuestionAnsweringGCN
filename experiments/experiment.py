class Experiment:

    model_updater = None
    model = None
    logger = None

    def __init__(self, model_trainer, model_tester, model, logger):
        self.model_trainer = model_trainer
        self.model_tester = model_tester
        self.model = model
        self.logger = logger

    def run(self):
        self.logger.write("Initializing model...", area="training", subject="initialization")
        self.model.initialize()
        self.logger.write("Training...", area="training", subject="start")
        self.model_trainer.train(self.model)
        self.logger.write("Testing...", area="testing", subject="start")
        for evaluation in self.model_tester.test(self.model, "test"):
            print(evaluation.summary)

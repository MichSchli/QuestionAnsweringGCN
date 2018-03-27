class Experiment:

    model_updater = None
    model = None

    def __init__(self, model_trainer, model_tester, model):
        self.model_trainer = model_trainer
        self.model_tester = model_tester
        self.model = model

    def run(self):
        self.model.initialize()
        self.model_trainer.train(self.model)
        summary = self.model_tester.test(self.model, "test").summary()
        print(summary)

class ModelTrainer:

    example_generator = None
    example_extender = None
    example_batcher = None
    model_updater = None

    max_iterations = None
    validate_every_n = None
    report_loss_every_n = None

    def __init__(self, example_generator, example_extender, example_batcher, model_updater, validation_evaluator):
        self.example_generator = example_generator
        self.example_extender = example_extender
        self.example_batcher = example_batcher
        self.model_updater = model_updater
        self.validation_evaluator = validation_evaluator

    def train(self, model):
        previous_score = 0

        for iteration in range(1,self.max_iterations+1):
            self.do_train_iteration(model)

            if iteration % self.validate_every_n == 0:
                evaluation = self.validation_evaluator.test(model, "valid")
                print(evaluation.summary())

                score = evaluation.macro_f1

                if self.early_stopping and score <= previous_score:
                    break

                previous_score = score

    def do_train_iteration(self, model):
        count_batches = 0
        for example in self.example_generator.iterate('train', shuffle=True):
            example = self.example_extender.extend(example)
            potential_batch = self.example_batcher.put_example(example)
            if potential_batch:
                loss = model.update(potential_batch)

                count_batches += 1

                if count_batches % self.report_loss_every_n == 0:
                    print(loss)

        last_batch = self.example_batcher.get_batch()
        if last_batch.has_examples():
            loss = model.update(last_batch)

            count_batches += 1

            if count_batches % self.report_loss_every_n == 0:
                print(loss)
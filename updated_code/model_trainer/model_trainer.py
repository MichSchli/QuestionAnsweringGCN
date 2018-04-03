class ModelTrainer:

    example_generator = None
    example_extender = None
    example_batcher = None
    model_updater = None

    max_iterations = None
    validate_every_n = None
    report_loss_every_n = None

    logger = None

    def __init__(self, example_generator, example_extender, example_batcher, validation_evaluator, logger):
        self.example_generator = example_generator
        self.example_extender = example_extender
        self.example_batcher = example_batcher
        self.validation_evaluator = validation_evaluator
        self.logger = logger

    def train(self, model):
        previous_score = 0

        for iteration in range(1,self.max_iterations+1):
            self.do_train_iteration(model)

            if iteration % self.validate_every_n == 0:
                self.logger.write("Validation at iteration "+str(iteration) + ":", area="training", subject="validation_start")
                evaluation = self.validation_evaluator.test(model, "valid")
                self.logger.write(evaluation.summary(), area="training", subject="validation_performance")

                score = evaluation.macro_f1

                if self.early_stopping and score <= previous_score:
                    self.logger.write("Stopping early at " + str(iteration) + ".", area="training",
                                      subject="early_stopping")

                previous_score = score

    def do_train_iteration(self, model):
        count_batches = 0
        loss_accumulator = 0.0
        for example in self.example_generator.iterate('train', shuffle=True):
            example = self.example_extender.extend(example)
            potential_batch = self.example_batcher.put_example(example)
            if potential_batch:
                loss = model.update(potential_batch)
                loss_accumulator += loss

                count_batches += 1

                if count_batches % self.report_loss_every_n == 0:
                    average_loss = loss_accumulator / self.report_loss_every_n
                    loss_accumulator = 0
                    self.logger.write("Loss at batch "+str(count_batches) + ": " + str(average_loss), area="training", subject="loss")

        last_batch = self.example_batcher.get_batch()
        if last_batch.has_examples():
            loss = model.update(last_batch)
            loss_accumulator += loss

            count_batches += 1

            if count_batches % self.report_loss_every_n == 0:
                average_loss = loss_accumulator / self.report_loss_every_n
                self.logger.write("Loss at batch "+str(count_batches) + ": " + str(average_loss), area="training", subject="loss")
class ModelTrainer:

    example_generator = None
    example_extender = None
    example_batcher = None
    model_updater = None
    max_iterations = None

    def __init__(self, example_generator, example_extender, example_batcher, model_updater):
        self.example_generator = example_generator
        self.example_extender = example_extender
        self.example_batcher = example_batcher
        self.model_updater = model_updater

    def train(self, model):
        for iteration in range(self.max_iterations):
             self.do_train_iteration(model)

    def do_train_iteration(self, model):
        for example in self.example_generator.iterate('train', shuffle=True):
            example = self.example_extender.extend(example, 'train')
            potential_batch = self.example_batcher.put_example(example)
            if potential_batch:
                self.model_updater.update(model, potential_batch)

        last_batch = self.example_batcher.get_batch()
        if last_batch.has_examples():
            self.model_updater.update(model, last_batch)
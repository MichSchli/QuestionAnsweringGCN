class ModelTester:

    example_generator = None
    example_extender = None
    example_batcher = None
    evaluator = None

    def __init__(self, example_generator, example_extender, example_batcher, evaluator):
        self.example_generator = example_generator
        self.example_extender = example_extender
        self.example_batcher = example_batcher
        self.evaluator = evaluator

    def test(self, model):
        self.evaluator.begin_evaluation()

        for example in self.example_generator.iterate('test', shuffle=False):
            example = self.example_extender.extend(example, 'test')
            self.example_batcher.put_example(example)
            potential_batch = self.example_batcher.put_example(example)

            if potential_batch:
                model.predict(potential_batch)
                for example in potential_batch.get_examples():
                    self.evaluator.add_prediction(example)

        return self.evaluator.final_scores()
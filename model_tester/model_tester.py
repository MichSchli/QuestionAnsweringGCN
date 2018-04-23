from models.prediction import Prediction


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

    def test(self, model, dataset):
        self.evaluator.begin_evaluation()

        for example in self.example_generator.iterate(dataset, shuffle=False):
            example = self.example_extender.extend(example)

            if not self.validate_example(example):
                self.add_dummy_prediction(example)
                self.evaluator.add_prediction(example)
                continue

            potential_batch = self.example_batcher.put_example(example)

            if potential_batch:
                model.predict(potential_batch)
                for example in potential_batch.get_examples():
                    self.evaluator.add_prediction(example)

        last_batch = self.example_batcher.get_batch()
        if last_batch.has_examples():
            model.predict(last_batch)
            for example in last_batch.get_examples():
                self.evaluator.add_prediction(example)

        return self.evaluator.final_scores()

    def validate_example(self, example):
        return len(example.mentions) > 0

    def add_dummy_prediction(self, example):
        prediction = Prediction()
        vertex_indexes = []
        vertex_labels = []

        prediction.add_predictions(vertex_indexes, vertex_labels)

        example.prediction = prediction

from models.prediction import Prediction


class ModelTester:

    example_generator = None
    example_extender = None
    example_batcher = None
    evaluator = None

    distribute_predictions = False

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
                self.predict(model, potential_batch)
                for example in potential_batch.get_examples():
                    self.evaluator.add_prediction(example)

        last_batch = self.example_batcher.get_batch()
        if last_batch.has_examples():
            self.predict(model, last_batch)
            for example in last_batch.get_examples():
                self.evaluator.add_prediction(example)

        return self.evaluator.final_scores()

    def predict(self, model, batch):
        predictions = model.predict_batch(batch)

        if self.distribute_predictions:
            example_begin_index = 0
            for example in batch.examples:
                scores = predictions[example_begin_index:example_begin_index + example.count_entities()]
                example_begin_index += example.count_entities()

                prediction = Prediction()
                vertex_indexes = example.graph.get_entity_vertices()
                vertex_labels = [example.graph.map_to_name_or_label(v) for v in vertex_indexes]

                prediction.add_predictions(vertex_labels, scores)
                example.prediction = prediction
        else:
            for i, example in enumerate(batch.examples):
                example.prediction = predictions[i]


    def validate_example(self, example):
        return not self.distribute_predictions or len(example.mentions) > 0

    def add_dummy_prediction(self, example):
        prediction = Prediction()
        vertex_indexes = []
        vertex_labels = []

        prediction.add_predictions(vertex_indexes, vertex_labels)

        example.prediction = prediction

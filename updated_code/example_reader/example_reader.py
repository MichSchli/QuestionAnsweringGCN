from example_reader.example import Example


class ExampleReader:
    question_reader = None

    def __init__(self, question_reader, graph_reader):
        self.question_reader = question_reader
        self.graph_reader = graph_reader

    def iterate(self, dataset, shuffle=False):
        for question in self.question_reader.iterate(dataset, shuffle=shuffle):
            example = Example()
            example.words = question[0]
            example.mentions = question[1]
            example.gold_answers = question[2]

            mention_entities = example.get_mentioned_entities()
            example.graph = self.graph_reader.get_neighborhood_graph(mention_entities)

            print(example.graph)

            yield example
from example_reader.example import Example
import random

class ExampleReader:

    question_reader = None
    graph_reader = None
    mention_reader = None
    gold_answer_reader = None
    project_names = False

    def __init__(self, question_reader, graph_reader, mention_reader, gold_answer_reader, dataset_map):
        self.question_reader = question_reader
        self.graph_reader = graph_reader
        self.dataset_map = dataset_map
        self.mention_reader = mention_reader
        self.gold_answer_reader = gold_answer_reader
        self.dataset = {}

    def iterate(self, dataset, shuffle=False):
        if dataset not in self.dataset:
            self.read_data(dataset)

        data = self.dataset[dataset]
        indexes = list(range(len(data[0])))
        if shuffle:
            random.shuffle(indexes)

        for i in indexes:
            example = Example()
            example.question = self.question_reader.build(data[0][i])
            example.mentions = self.mention_reader.build(data[1][i])

            if dataset == "train" and len(example.mentions) == 0:
                continue

            example.graph = self.graph_reader.get_neighborhood_graph(example.get_mentioned_entities())
            example.index_mentions()
            example.gold_answers = self.gold_answer_reader.build(data[2][i])
            example.index_gold_answers(self.project_names)

            yield example

    def read_data(self, dataset):
        data = [[[]], [], []]
        with open(self.dataset_map[dataset]) as data_file:

            mode = 0

            for line in data_file:
                line = line.strip()

                if line:
                    data[mode][-1].append(line.split('\t'))
                else:
                    mode = (mode + 1) % 3
                    data[mode].append([])
        self.dataset[dataset] = data
        return data
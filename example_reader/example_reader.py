from example_reader.example import Example
import random

class ExampleReader:

    question_reader = None
    graph_reader = None
    mention_reader = None
    gold_answer_reader = None
    project_names = False
    deplambda = None
    deplambda_map = {"train": "data/webquestions/siva.train.split.deplambda.conll"}

    def __init__(self, question_reader, graph_reader, mention_reader, gold_answer_reader, gold_path_reader, dataset_map, deplambda_map = None, deplambda=False):
        self.question_reader = question_reader
        self.graph_reader = graph_reader
        self.dataset_map = dataset_map
        self.mention_reader = mention_reader
        self.gold_answer_reader = gold_answer_reader
        self.dataset = {}
        self.deplambda = deplambda
        self.gold_path_reader = gold_path_reader
        #self.deplambda_map = deplambda_map

    def iterate(self, dataset, shuffle=False):
        if dataset not in self.dataset:
            self.read_data(dataset)

        data = self.dataset[dataset]
        indexes = list(range(len(data[0])))
        if shuffle:
            random.shuffle(indexes)

        for i in indexes:
            example = Example()
            example.project_names = self.project_names
            example.question = self.question_reader.build(data[0][i])
            example.mentions = self.mention_reader.build(data[1][i])

            if dataset == "train" and len(example.mentions) == 0:
                continue

            example.graph = self.graph_reader.get_neighborhood_graph(example.get_mentioned_entities())
            example.index_mentions()
            example.gold_answers = self.gold_answer_reader.build(data[2][i])
            example.index_gold_answers()

            example.gold_paths = self.gold_path_reader.find(i, dataset)

            yield example

    def read_data(self, dataset):
        data = [[[]], [], []]
        with open(self.dataset_map[dataset], encoding="utf8") as data_file:

            mode = 0

            for line in data_file:
                line = line.strip()

                if line:
                    data[mode][-1].append(line.split('\t'))
                else:
                    mode = (mode + 1) % 3
                    data[mode].append([])


        if self.deplambda:
            deplambda_data = self.read_deplambda(dataset)
            for i in range(len(data)):
                data[i].append(deplambda_data[i])

        self.dataset[dataset] = data
        return data

    def read_deplambda(self, dataset):
        with open(self.deplambda_map[dataset]) as data_file:
            mode = None
            data = []
            for line in data_file:
                line = line.strip()

                if line.startswith("SENTENCE:"):
                    data.append([[[], []]])
                    mode = 0
                elif not line and mode == 0:
                    mode = 1
                elif line and mode == 1:
                    data[-1][-1][0].append(line.split("\t"))
                elif not line and mode == 1:
                    mode = 2
                elif line and mode == 2:
                    data[-1][-1][1].append(line.split("\t"))
                elif not line and mode == 2:
                    mode = 3
                elif line and mode == 3:
                    data[-1].append([[], []])
                    data[-1][-1][0].append(line.split("\t"))
                    mode = 1
                elif not line and mode == 3:
                    mode = 4

            return data




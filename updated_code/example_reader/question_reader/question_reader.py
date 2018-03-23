import random


class QuestionReader:

    dataset_map = None
    dataset = None

    def __init__(self, dataset_map):
        self.dataset_map = dataset_map
        self.dataset = {}

    def iterate(self, dataset, shuffle=False):
        if dataset not in self.dataset:
            self.read_data(dataset)

        data = self.dataset[dataset]
        indexes = list(range(len(data[0])))
        if shuffle:
            random.shuffle(indexes)

        for i in indexes:
            yield [data[j][i] for j in range(len(data))]

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



from example_reader.gold_path_reader.gold_path import GoldPath


class GoldPathFinder:

    datasets = None

    def __init__(self, dataset_map):
        self.datasets = {d : self.load(f) for d,f in dataset_map.items()}

    def find(self, example):
        gold_indexes = example.get_gold_indexes()

        gold_paths = [example.graph.entity_centroid_paths[i] for i in gold_indexes]

        print(example)
        print(gold_paths)
        exit()

    def load(self, dataset):
        f = open(dataset, "r")

        dict = {}
        counter = -1

        for line in f:
            line = line.strip()
            if not line:
                continue


            if line.startswith("SENTENCE:"):
                counter += 1
                dict[counter] = []
            else:
                items = line.split("\t")
                gold_path = GoldPath()
                gold_path.is_simple = items[0].endswith(".1") or items[0].endswith(".2")

                gold_path.relation_mention_inverse = items[0].endswith(".inverse") if not gold_path.is_simple else items[0].endswith(".2")
                gold_path.relation_gold_inverse = items[1].endswith(".inverse") if not gold_path.is_simple else items[0].endswith(".2")

                gold_path.relation_mention = items[0].replace(".inverse", "") if not gold_path.is_simple else items[0][:-2]
                gold_path.relation_gold = items[1].replace(".inverse", "") if not gold_path.is_simple else items[1][:-2]

                gold_path.score = float(items[2])

                dict[counter].append(gold_path)

        return dict
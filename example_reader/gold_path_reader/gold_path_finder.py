from example_reader.gold_path_reader.gold_path import GoldPath
import numpy as np


class GoldPathFinder:

    datasets = None
    relation_index = None

    def __init__(self, relation_index, dataset_map):
        self.datasets = {d : self.load(f) for d,f in dataset_map.items()}
        self.relation_index = relation_index

    def find(self, index, dataset):
        gold_paths = self.datasets[dataset][index]

        for g in gold_paths:
            l_path = "http://rdf.freebase.com/ns/" + g.relation_mention + " | http://rdf.freebase.com/ns/" + g.relation_gold
            l_index = self.relation_index.index(l_path)

            #if g.relation_mention_inverse :
            #    l_index += self.relation_index.vector_count

            vector = np.zeros((self.relation_index.vector_count))
            vector[l_index] = 1
            g.vector = vector

        return gold_paths

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

                gold_path.relation_mention = items[0]
                gold_path.relation_gold = items[1]

                gold_path.score = float(items[2])

                dict[counter].append(gold_path)

        return dict
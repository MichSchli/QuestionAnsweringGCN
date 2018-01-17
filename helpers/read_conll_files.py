import json
import numpy as np
import random


class ConllReader:

    output=None
    entity_prefix=None
    max_elements = None

    def __init__(self, filename, entity_prefix="", max_elements=None, greedy_disambiguation=False):
        self.filename = filename
        self.entity_prefix = entity_prefix
        self.max_elements = None
        self.greedy_disambiguation = greedy_disambiguation

    def iterate(self, output=None, shuffle=False):
        dictionary = {}

        with open(self.filename) as data_file:

            sentence_matrix = []
            gold_matrix = []
            entity_matrix = []

            reading_sentence = True
            reading_entities = False
            counter = 0

            dicts = []

            for line in data_file:
                line = line.strip()

                if line and reading_sentence:
                    sentence_matrix.append(line.split('\t'))
                elif line and reading_entities:
                    entity_matrix.append(line.split('\t'))
                elif line and not reading_sentence and not reading_entities:
                    gold_matrix.append(line.split('\t'))
                elif not line and reading_sentence:
                    reading_sentence = False
                    reading_entities = True
                elif not line and reading_entities:
                    reading_entities = False
                elif not line and not reading_sentence and not reading_entities:
                    reading_sentence = True

                    yield from self.yield_example(dictionary, dicts, entity_matrix, gold_matrix, sentence_matrix,
                                                  shuffle)

                    counter += 1
                    if self.max_elements is not None and counter == self.max_elements:
                        break

                    dictionary = {}

                    sentence_matrix = []
                    entity_matrix = []
                    gold_matrix = []

            if not reading_sentence and not reading_entities:

                yield from self.yield_example(dictionary, dicts, entity_matrix, gold_matrix, sentence_matrix,
                                              shuffle)


            random.shuffle(dicts)
            for dict in dicts:
                yield dict

    def yield_example(self, dictionary, dicts, entity_matrix, gold_matrix, sentence_matrix, shuffle):

        if self.greedy_disambiguation and len(entity_matrix) > 0:
            entity_matrix = [entity_matrix[0]]


        dictionary["mentioned_entities"] = np.unique(
            np.array([self.entity_prefix + entry[2] for entry in entity_matrix]))
        dictionary["sentence"] = sentence_matrix
        dictionary["sentence_entity_map"] = np.array(
            [[entry[0], entry[1], self.entity_prefix + entry[2], entry[3]] for entry in entity_matrix])
        dictionary["gold_entities"] = np.array([e[0] if e[0] != "_" else e[1] for e in gold_matrix])
        if shuffle:
            dicts.append(dictionary)
        else:
            yield dictionary

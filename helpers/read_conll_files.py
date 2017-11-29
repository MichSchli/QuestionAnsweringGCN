import json
import numpy as np


class ConllReader:

    output=None
    entity_prefix=None
    max_elements = None

    def __init__(self, filename, entity_prefix="", max_elements=None):
        self.filename = filename
        self.entity_prefix = entity_prefix
        self.max_elements = max_elements

    def iterate(self, output=None):
        dictionary = {}

        with open(self.filename) as data_file:

            sentence_matrix = []
            gold_matrix = []
            entity_matrix = []

            reading_sentence = True
            reading_entities = False
            counter = 0

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

                    dictionary["mentioned_entities"] = np.unique(np.array([self.entity_prefix + entry[2] for entry in entity_matrix]))
                    dictionary["sentence"] = sentence_matrix
                    dictionary["sentence_entity_map"] = np.array([[entry[0], entry[1], self.entity_prefix + entry[2], entry[3]] for entry in entity_matrix])
                    dictionary["gold_entities"] = np.array([e[0] if e[0] != "_" else e[1] for e in gold_matrix])

                    yield dictionary

                    counter += 1
                    if self.max_elements is not None and counter == self.max_elements:
                        break

                    """if output == "entities":
                        sentence_entities = np.unique(np.array([self.entity_prefix + entry[2] for entry in entity_matrix]))
                        yield sentence_entities
                    elif output == "gold":
                        gold_entities = np.array([e[0] if e[0] != "_" else e[1] for e in gold_matrix])
                        yield gold_entities
                    elif output == "sentences":
                        yield sentence_matrix
                    elif output == "sentences+entities":
                        entity_matrix = np.array([[entry[0], entry[1], self.entity_prefix + entry[2], entry[3]] for entry in entity_matrix])
                        yield sentence_matrix, entity_matrix
                    else:
                        yield sentence_matrix, gold_matrix

                    """

                    dictionary = {}

                    sentence_matrix = []
                    entity_matrix = []
                    gold_matrix = []

import json
import numpy as np


class ConllReader:

    output=None
    entity_prefix=None

    def __init__(self, output=None, entity_prefix="http://rdf.freebase.com/ns/"):
        self.output=output
        self.entity_prefix = entity_prefix

    def parse_file(self, filename, output=None):
        if output == None:
            output = self.output

        with open(filename) as data_file:
            sentence_matrix = []
            gold_matrix = []
            entity_matrix = []

            reading_sentence = True
            reading_entities = False
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

                    if output == "entities":
                        sentence_entities = np.unique(np.array([self.entity_prefix + entry[2] for entry in entity_matrix]))
                        yield sentence_entities
                    elif output == "gold":
                        gold_entities = np.array([e[0] if e[0] != "_" else e[1] for e in gold_matrix])
                        yield gold_entities
                    else:
                        yield sentence_matrix, gold_matrix

                    sentence_matrix = []
                    gold_matrix = []

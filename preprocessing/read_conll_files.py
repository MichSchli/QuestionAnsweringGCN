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

            reading_sentence = True
            for line in data_file:
                line = line.strip()

                if line and reading_sentence:
                    sentence_matrix.append(line.split('\t'))
                elif line and not reading_sentence:
                    gold_matrix.append(line.split('\t'))
                elif not line and reading_sentence:
                    reading_sentence = False
                elif not line and not reading_sentence:
                    reading_sentence = True

                    if output == "entities":
                        sentence_entities = ",".join([s[6] for s in sentence_matrix if s[6] != "_"]).split(",")
                        sentence_entities = np.unique(np.array([self.entity_prefix + entity for entity in sentence_entities]))
                        yield sentence_entities
                    elif output == "gold":
                        gold_entities = [e[0] if e[0] != "_" else e[1] for e in gold_matrix]
                        yield gold_entities
                    else:
                        yield sentence_matrix, gold_matrix

                    sentence_matrix = []
                    gold_matrix = []

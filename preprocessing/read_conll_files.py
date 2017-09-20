import json
import numpy as np


class ConllReader:

    entity_prefix=None

    def __init__(self, entity_prefix="http://rdf.freebase.com/ns/"):
        self.entity_prefix = entity_prefix

    def parse_file(self, filename):

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
                    yield sentence_matrix, gold_matrix

                    sentence_matrix = []
                    gold_matrix = []

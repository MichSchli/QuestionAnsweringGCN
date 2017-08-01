import json
import numpy as np


class JsonReader:

    output=None

    def __init__(self, output=None):
        self.output=output

    def parse_json_to_gold(self, json_line):
        freebase_entities = np.array(json_line['answerSubset'])
        return freebase_entities

    def parse_file(self, filename, output=None):
        if output is None:
            output = self.output

        with open(filename) as data_file:
            for line in (data_file):
                line = json.loads(line)
                if output == "gold":
                    yield self.parse_json_to_gold(line)
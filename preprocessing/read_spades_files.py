import json
import numpy as np


class JsonReader:

    output=None

    def __init__(self, output=None):
        self.output=output

    def parse_json_to_gold(self, json_line):
        freebase_entities = np.array(["http://rdf.freebase.com/ns/"+e for e in json_line['answerSubset']])
        return freebase_entities

    def parse_json_to_entities(self, json_line):
        return np.array(["http://rdf.freebase.com/ns/"+entity["entity"] for entity in json_line['entities']])

    def parse_file(self, filename, output=None):
        if output is None:
            output = self.output

        with open(filename) as data_file:
            for line in (data_file):
                line = json.loads(line)
                if output == "gold":
                    yield self.parse_json_to_gold(line)
                elif output == "entities":
                    yield self.parse_json_to_entities(line)
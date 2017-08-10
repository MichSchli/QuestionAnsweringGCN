import json
import numpy as np


class JsonReader:

    output=None
    entity_prefix=None

    def __init__(self, output=None, entity_prefix="http://rdf.freebase.com/ns/"):
        self.output=output
        self.entity_prefix = entity_prefix

    def parse_json_to_gold(self, json_line):
        freebase_entities = np.array([self.entity_prefix+e for e in json_line['answerSubset']])
        return freebase_entities

    def parse_json_to_entities(self, json_line):
        return np.array([self.entity_prefix+entity["entity"] for entity in json_line['entities']])

    def parse_file(self, filename, output=None, print_progress=False):
        if output is None:
            output = self.output

        if print_progress:
            with open(filename) as f:
                total_lines = sum(1 for _ in f)
                print("")

        with open(filename) as data_file:
            for i,line in enumerate(data_file):
                if print_progress:
                    print("\x1b[1A\rParsing " + str(i) + "/" + str(total_lines))
                line = json.loads(line)
                if output == "gold":
                    yield self.parse_json_to_gold(line)
                elif output == "entities":
                    #print(self.parse_json_to_entities(line))
                    yield self.parse_json_to_entities(line)

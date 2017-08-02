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
                    yield self.parse_json_to_entities(line)

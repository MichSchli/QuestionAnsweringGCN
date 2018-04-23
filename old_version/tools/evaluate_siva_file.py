from evaluation.python_evaluator import Evaluator
from helpers.read_conll_files import ConllReader
import json
import numpy as np


class SivaJsonReader():

    gold = None

    def __init__(self, gold=False):
        self.gold = gold

    def parse_file(self, filename):
        for line in open("data/webquestions/wq.bow.graphs.train.json"):
            json_parse = json.loads(line.strip())
            answer = json_parse["answer"]

            if self.gold:
                yield answer
            else:
                forest = json_parse["forest"]
                candidates = [graph["denotation"] for graph in forest[0]["graphs"]]

                if len(candidates) == 0:
                    yield []
                    continue

                overlap = np.array([np.isin(candidate, answer).sum() for candidate in candidates])
                best = overlap.argmax()
                yield candidates[best]

gold_reader = SivaJsonReader(gold=True)
siva_reader = SivaJsonReader(gold=False)


evaluator = Evaluator(siva_reader, gold_reader)
evaluator.parse_file("data/webquestions/train.internal.conll", method="macro")
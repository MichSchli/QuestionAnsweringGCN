import argparse

from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface
from baselines.oracle_candidate import OracleCandidate
from baselines.random_single_candidate import RandomSingleCandidate
from grounding.json_to_candidate_neighborhood import CandidateNeighborhoodGenerator
from preprocessing.read_spades_files import JsonReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

gold_reader = JsonReader(output="gold")

freebase = FreebaseInterface()
sentence_reader = JsonReader()
candidate_generator = CandidateNeighborhoodGenerator(freebase, sentence_reader)
gold_reader_for_oracle = JsonReader(output="gold")

strategy = OracleCandidate(candidate_generator, gold_reader_for_oracle)

gold_iterator = gold_reader.parse_file(args.file)
prediction_iterator = strategy.parse_file(args.file)

for pred, gold in zip(prediction_iterator, gold_iterator):
    print("Made prediction: " + str(pred) + " || Gold: "+ str(gold))
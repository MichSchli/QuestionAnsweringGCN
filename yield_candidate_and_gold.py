import argparse

from baselines.oracle_candidate import OracleCandidate
from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from database_interface.properties.vertex_property_retriever import VertexPropertyRetriever
from grounding.json_to_candidate_neighborhood import CandidateNeighborhoodGenerator
from preprocessing.read_spades_files import JsonReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

gold_reader = JsonReader(output="gold")

#database_interface = #FreebaseInterface()
database_interface = CsvInterface("data/toy/toy.graph")
database = HypergraphInterface(database_interface, AllThroughExpansionStrategy(), VertexPropertyRetriever(database_interface))
sentence_reader = JsonReader()
candidate_generator = CandidateNeighborhoodGenerator(database, sentence_reader, neighborhood_search_scope=1)
gold_reader_for_oracle = JsonReader(output="gold")

strategy = OracleCandidate(candidate_generator, gold_reader_for_oracle)

gold_iterator = gold_reader.parse_file(args.file)
prediction_iterator = strategy.parse_file(args.file)

for pred, gold in zip(prediction_iterator, gold_iterator):
    print("Made prediction: " + str(pred) + " || Gold: "+ str(gold))
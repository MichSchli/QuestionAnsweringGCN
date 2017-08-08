import argparse

from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface
from baselines.oracle_candidate import OracleCandidate
from baselines.random_single_candidate import RandomSingleCandidate
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from database_interface.properties.vertex_property_retriever import VertexPropertyRetriever
from evaluation.python_evaluator import Evaluator
from grounding.json_to_candidate_neighborhood import CandidateNeighborhoodGenerator
from preprocessing.read_spades_files import JsonReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

gold_reader = JsonReader(output="gold")

database_interface = FreebaseInterface()
#database_interface = CsvInterface("data/toy/toy.graph")
database = HypergraphInterface(database_interface, AllThroughExpansionStrategy(), VertexPropertyRetriever(database_interface))
sentence_reader = JsonReader()
candidate_generator = CandidateNeighborhoodGenerator(database, sentence_reader)
gold_reader_for_oracle = JsonReader(output="gold")

strategy = OracleCandidate(candidate_generator, gold_reader_for_oracle)

evaluator = Evaluator(strategy, gold_reader)
evaluator.parse_file(args.file)
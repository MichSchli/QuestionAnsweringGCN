import argparse

from auxilliary_models.aux_parser_wrapper import AuxParserWrapper
from candidate_generation.neighborhood_candidate_generator import NeighborhoodCandidateGenerator
from candidate_selection.baselines.oracle_candidate import OracleCandidate
from candidate_selection.models.dumb_entity_embedding_vs_bag_of_words import DumbEntityEmbeddingVsBagOfWords
from candidate_selection.models.lstm_vs_gcn import LstmVsGcnModel
from candidate_selection.tensorflow_candidate_selector import TensorflowCandidateSelector
from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.expansion_strategies.only_freebase_element_strategy import OnlyFreebaseExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from evaluation.python_evaluator import Evaluator
from facts.database_facts.csv_facts import CsvFacts
from facts.database_facts.freebase_facts import FreebaseFacts
from helpers.read_conll_files import ConllReader
from model_construction.model_builder import ModelBuilder

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--algorithm', type=str, help='The name of the algorithm to be tested')
parser.add_argument('--train_file', type=str, help='The location of the .conll-file to be used for training')
parser.add_argument('--test_file', type=str, help='The location of the .conll file to be used for validation')
parser.add_argument('--version', type=str, help='Optional version number')
parser.add_argument('--save', type=bool, default=False, help='Determines whether to save the best model on disk.')
args = parser.parse_args()

version_string = ".v" + args.version if args.version else ""

algorithm_name = ".".join(args.algorithm.split("/")[-1].split(".")[:-1])
log_file_location = "logs/" + algorithm_name + version_string + ".txt"
save_file_location = "stored_models/" + algorithm_name + version_string + ".ckpt"

model_builder = ModelBuilder()
model_builder.build(args.algorithm, version=args.version)

gold_reader = ConllReader(args.valid_file)
evaluator = Evaluator(gold_reader)

model = model_builder.first()
train_file_iterator = ConllReader(args.train_file)
model.train(train_file_iterator)

valid_file_iterator = ConllReader(args.valid_file)
prediction = model.predict(valid_file_iterator)
evaluation = evaluator.evaluate(prediction)

evaluation.pretty_print()
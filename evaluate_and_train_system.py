import argparse

from auxilliary_models.aux_parser_wrapper import AuxParserWrapper
from candidate_generation.neighborhood_candidate_generator import NeighborhoodCandidateGenerator
from candidate_selection.baselines.oracle_candidate import OracleCandidate
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

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--train_file', type=str, help='The location of the .conll-file to be used for training')
parser.add_argument('--test_file', type=str, help='The location of the .conll file to be used for testing')
parser.add_argument('--backend', type=str, help='The location of the .graph file to be used as backend, or left blank for Freebase')
args = parser.parse_args()

gold_reader = ConllReader(output="gold")

if args.backend == "freebase":
    database_interface = FreebaseInterface()
    expansion_strategy = OnlyFreebaseExpansionStrategy()
    entity_prefix = "http://rdf.freebase.com/ns/"
    facts = FreebaseFacts()
else:
    database_interface = CsvInterface(args.backend)
    expansion_strategy = AllThroughExpansionStrategy()
    entity_prefix = ""
    facts = CsvFacts(args.backend)

database = HypergraphInterface(database_interface, expansion_strategy)
sentence_reader = ConllReader(entity_prefix=entity_prefix)
candidate_generator = NeighborhoodCandidateGenerator(database, sentence_reader, neighborhood_search_scope=1, extra_literals=True)

# Move this to argument/config later:
strategy_string = "lstm+gcn"
if strategy_string == "oracle":
    gold_reader_for_oracle = ConllReader(output="gold", entity_prefix=entity_prefix)
    strategy = OracleCandidate(candidate_generator, gold_reader_for_oracle)
elif strategy_string == "lstm+gcn":
    gold_reader_for_training = ConllReader(output="gold", entity_prefix=entity_prefix)
    sentence_reader = ConllReader(output="sentences+entities", entity_prefix=entity_prefix)
    aux_read_wrapper = AuxParserWrapper(sentence_reader, args.train_file)
    model = LstmVsGcnModel(facts, 100, aux_read_wrapper)
    strategy = TensorflowCandidateSelector(model, candidate_generator, gold_reader_for_training, facts)

strategy.train(args.train_file)

if strategy_string == "lstm+gcn":
    aux_read_wrapper.filename = args.test_file

evaluator = Evaluator(strategy, gold_reader)
evaluator.parse_file(args.test_file)

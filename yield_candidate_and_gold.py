import argparse

from additional_graphs.toy_additional_graphs import ToyAdditionalGraphs
from candidate_selection.baselines.oracle_candidate import OracleCandidate
from candidate_selection.models.candidate_GCN_only import CandidateGcnOnlyModel
from candidate_selection.models.candidate_and_aux_GCN import CandidateAndAuxGcnModel
from candidate_selection.tensorflow_candidate_selector import TensorflowCandidateSelector
from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from database_interface.properties.hypergraph_vertex_property_retriever import HypergraphPropertyRetriever
from database_interface.properties.vertex_property_retriever import VertexPropertyRetriever
from facts.database_facts.csv_facts import CsvFacts
from facts.database_facts.freebase_facts import FreebaseFacts
from grounding.json_to_candidate_neighborhood import CandidateNeighborhoodGenerator
from preprocessing.read_spades_files import JsonReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

gold_reader = JsonReader(output="gold", entity_prefix="http://rdf.freebase.com/ns/")

#facts = FreebaseFacts()
facts = CsvFacts("data/toy/toy.graph")

#database_interface = FreebaseInterface()
database_interface = CsvInterface("data/toy/toy.graph")
database = HypergraphInterface(database_interface, AllThroughExpansionStrategy(), HypergraphPropertyRetriever(VertexPropertyRetriever(database_interface)))
sentence_reader = JsonReader(entity_prefix="http://rdf.freebase.com/ns/")
candidate_generator = CandidateNeighborhoodGenerator(database, sentence_reader, neighborhood_search_scope=1)

gold_reader_for_training = JsonReader(output="gold", entity_prefix="http://rdf.freebase.com/ns/")
aux_iterator = ToyAdditionalGraphs().produce_additional_graphs()
model = CandidateAndAuxGcnModel(facts, aux_iterator)
strategy = TensorflowCandidateSelector(model, candidate_generator, gold_reader_for_training, facts)

gold_iterator = gold_reader.parse_file(args.file)
prediction_iterator = strategy.parse_file(args.file)

for pred, gold in zip(prediction_iterator, gold_iterator):
    print("Made prediction: " + str(pred) + " || Gold: "+ str(gold))

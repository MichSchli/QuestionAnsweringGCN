from candidate_generation.candidate_generator_cache import CandidateGeneratorCache
from candidate_generation.neighborhood_candidate_generator import NeighborhoodCandidateGenerator
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.expansion_strategies.only_freebase_element_strategy import OnlyFreebaseExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from helpers.read_conll_files import ConllReader

database_interface = FreebaseInterface()
expansion_strategy = OnlyFreebaseExpansionStrategy()

prefix = "http://rdf.freebase.com/ns/"
disk_cache = "/datastore/michael_cache/webquestions.1neighbors.cache"

database = HypergraphInterface(database_interface, expansion_strategy, prefix=prefix)

candidate_generator = NeighborhoodCandidateGenerator(database, neighborhood_search_scope=1,
                                                     extra_literals=True)
candidate_generator = CandidateGeneratorCache(candidate_generator,
                                              disk_cache=disk_cache)

train_file_iterator = ConllReader("data/webquestions/train.split.conll")
epoch_iterator = train_file_iterator.iterate()
epoch_iterator = candidate_generator.enrich(epoch_iterator)

for example in epoch_iterator:
    for edge in example["neighborhood"].entity_to_event_edges:
        print(edge[1])

    for edge in example["neighborhood"].event_to_entity_edges:
        print(edge[1])

    for edge in example["neighborhood"].entity_to_entity_edges:
        print(edge[1])
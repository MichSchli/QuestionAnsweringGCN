from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.expansion_strategies.only_freebase_element_strategy import OnlyFreebaseExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from database_interface.indexed_interface import IndexedInterface
from experiment_construction.candidate_generator_construction.candidate_generator_cache import CandidateGeneratorCache
from experiment_construction.candidate_generator_construction.neighborhood_candidate_generator import \
    NeighborhoodCandidateGenerator


class CandidateGeneratorFactory:

    def construct_candidate_generator(self, index, settings):
        if "prefix" in settings["endpoint"]:
            prefix = settings["endpoint"]["prefix"]
        else:
            prefix = ""

        if settings["endpoint"]["type"] == "sparql":
            database_interface = FreebaseInterface()
            expansion_strategy = OnlyFreebaseExpansionStrategy()
        elif settings["endpoint"]["type"] == "csv":
            database_interface = CsvInterface(settings["endpoint"]["file"])
            expansion_strategy = AllThroughExpansionStrategy()
            prefix = ""

        database = HypergraphInterface(database_interface, expansion_strategy, prefix=prefix)
        database = IndexedInterface(database, index.entity_indexer, index.relation_indexer)
        candidate_generator = NeighborhoodCandidateGenerator(database, neighborhood_search_scope=1,
                                                             extra_literals=True)

        disk_cache = settings["endpoint"]["disk_cache"] if "disk_cache" in settings["endpoint"] else None
        if disk_cache:
            candidate_generator = CandidateGeneratorCache(candidate_generator, disk_cache=disk_cache)

        #candidate_generator = NameMapper(candidate_generator)


        return candidate_generator

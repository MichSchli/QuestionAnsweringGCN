from example_reader.graph_reader.database_interface.data_interface.CsvInterface import CsvInterface
from example_reader.graph_reader.database_interface.data_interface.FreebaseInterface import FreebaseInterface
from example_reader.graph_reader.database_interface.expansion_strategies.all_through_expansion_strategy import \
    AllThroughExpansionStrategy
from example_reader.graph_reader.database_interface.expansion_strategies.only_freebase_element_strategy import \
    OnlyFreebaseExpansionStrategy
from example_reader.graph_reader.database_interface.hypergraph_interface import HypergraphInterface
from example_reader.graph_reader.empty_graph_reader import EmptyGraphReader
from example_reader.graph_reader.graph_cache import GraphCache
from example_reader.graph_reader.graph_converter import GraphConverter
from example_reader.graph_reader.graph_indexer import GraphIndexer
from example_reader.new_graph_reader.new_graph_reader import NewGraphReader


class GraphReaderFactory:

    def __init__(self, index_factory):
        self.index_factory = index_factory

    def get(self, experiment_configuration):
        if "prefix" in experiment_configuration["endpoint"]:
            prefix = experiment_configuration["endpoint"]["prefix"]
        else:
            prefix = ""

        if experiment_configuration["endpoint"]["type"] == "sparql":
            database_interface = FreebaseInterface()
        elif experiment_configuration["endpoint"]["type"] == "csv":
            database_interface = CsvInterface(experiment_configuration["endpoint"]["file"])
            expansion_strategy = AllThroughExpansionStrategy()
            prefix = ""

        if "no_graph_features" in experiment_configuration["other"] and experiment_configuration["other"]["no_graph_features"] == "True":
            return EmptyGraphReader()

        if "prefix" in experiment_configuration["endpoint"]:
            prefix = experiment_configuration["endpoint"]["prefix"]
        else:
            prefix = ""

        if experiment_configuration["endpoint"]["type"] == "sparql":
            database_interface = FreebaseInterface()
            expansion_strategy = OnlyFreebaseExpansionStrategy()
        elif experiment_configuration["endpoint"]["type"] == "csv":
            database_interface = CsvInterface(experiment_configuration["endpoint"]["file"])
            expansion_strategy = AllThroughExpansionStrategy()
            prefix = ""

        database_interface = HypergraphInterface(database_interface, expansion_strategy, prefix=prefix)
        graph_converter = GraphConverter(database_interface)
        graph_indexer = GraphIndexer(graph_converter,
                                     self.index_factory.get("vertices", experiment_configuration),
                                     self.index_factory.get("relations", experiment_configuration),
                                     self.index_factory.get("relation_parts", experiment_configuration))

        disk_cache = experiment_configuration["endpoint"]["disk_cache"] if "disk_cache" in experiment_configuration["endpoint"] else None
        if disk_cache:
            graph_cache = GraphCache(graph_indexer, disk_cache)


        return graph_cache
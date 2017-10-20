from candidate_generation.candidate_generator_cache import CandidateGeneratorCache
from candidate_generation.neighborhood_candidate_generator import NeighborhoodCandidateGenerator
from candidate_selection.baselines.oracle_candidate import OracleCandidate
from candidate_selection.models.dumb_entity_embedding_vs_bag_of_words import DumbEntityEmbeddingVsBagOfWords
from candidate_selection.models.dumb_entity_embedding_vs_lstm import DumbEntityEmbeddingVsLstm
from candidate_selection.tensorflow_candidate_selector import TensorflowCandidateSelector
from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.entity_cache_interface import EntityCacheInterface
from database_interface.expansion_strategies.all_through_expansion_strategy import AllThroughExpansionStrategy
from database_interface.expansion_strategies.only_freebase_element_strategy import OnlyFreebaseExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from facts.database_facts.csv_facts import CsvFacts
from facts.database_facts.freebase_facts import FreebaseFacts
from helpers.read_conll_files import ConllReader
from model_construction.settings_reader import SettingsReader
import itertools

class ModelBuilder:

    settings_reader = None
    settings = None

    def __init__(self):
        self.settings_reader = SettingsReader()

    def build(self, settings, version=None):
        self.settings = settings
        return self.create_model()

    def create_model(self):
        model_class = self.retrieve_class_name(self.settings["algorithm"]["model"]["stack_name"])
        model = model_class()
        model = self.wrap_model(model)
        for k, v in self.settings["algorithm"]["model"].items():
            if k == "stack_name":
                continue
            else:
                model.update_setting(k, v)
        return model

    def wrap_model(self, model):
        if "prefix" in self.settings["dataset"]["database"]:
            prefix = self.settings["dataset"]["database"]["prefix"]
        else:
            prefix = ""

        if self.settings["dataset"]["database"]["endpoint"] == "sparql":
            database_interface = FreebaseInterface()
            expansion_strategy = OnlyFreebaseExpansionStrategy()
            facts = FreebaseFacts()
        elif self.settings["dataset"]["database"]["endpoint"] == "csv":
            database_interface = CsvInterface(self.settings["dataset"]["database"]["file"])
            expansion_strategy = AllThroughExpansionStrategy()
            facts = CsvFacts(self.settings["dataset"]["database"]["file"])
            prefix = ""

        database = HypergraphInterface(database_interface, expansion_strategy, prefix=prefix)
        database = EntityCacheInterface(database)
        disk_cache = self.settings["dataset"]["database"]["disk_cache"] if "disk_cache" in self.settings["dataset"]["database"] else None
        candidate_generator = NeighborhoodCandidateGenerator(database, neighborhood_search_scope=1,
                                                             extra_literals=True)
        candidate_generator = CandidateGeneratorCache(candidate_generator,
                                                      disk_cache=disk_cache)

        # Should be refactored
        if model.is_tensorflow:
            candidate_selector = TensorflowCandidateSelector(model, candidate_generator)
        else:
            candidate_selector = model
            candidate_selector.set_neighborhood_generate(candidate_generator)

        candidate_selector.update_setting("facts", facts)
        return candidate_selector

    def first(self):
        return self.search().__next__()

    def search(self):
        configurations = []

        if "searchable" not in self.settings["algorithm"]:
            model = self.create_model()
            model.initialize()
            yield model
            return

        for k,v in self.settings["algorithm"]["searchable"].items():
            options = [(k, option) for option in v.split(",")]
            configurations.append(options)

        for combination in itertools.product(*configurations):
            model = self.create_model()
            for k,v in combination:
                model.update_setting(k,v)
            model.initialize()

            parameter_string = ",".join([k+"="+v for k,v in combination])
            yield parameter_string, model


    def retrieve_class_name(self, stack_name):
        if stack_name == "oracle":
            return OracleCandidate

        if stack_name == "bow+dumb":
            return DumbEntityEmbeddingVsBagOfWords

        if stack_name == "lstm+dumb":
            return DumbEntityEmbeddingVsLstm

        return None
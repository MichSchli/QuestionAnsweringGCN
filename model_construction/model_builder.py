from candidate_generation.neighborhood_candidate_generator import NeighborhoodCandidateGenerator
from candidate_selection.models.dumb_entity_embedding_vs_bag_of_words import DumbEntityEmbeddingVsBagOfWords
from candidate_selection.tensorflow_candidate_selector import TensorflowCandidateSelector
from database_interface.data_interface.CsvInterface import CsvInterface
from database_interface.data_interface.FreebaseInterface import FreebaseInterface
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

    def build(self, settings_file, version=None):
        self.settings = self.settings_reader.read(settings_file)
        return self.create_model()

    def create_model(self):
        model_class = self.retrieve_class_name(self.settings["model"]["stack_name"])
        model = model_class()
        model = self.wrap_model(model)
        for k, v in self.settings["model"].items():
            if k == "stack_name":
                continue
            else:
                model.update_setting(k, v)
        return model

    def wrap_model(self, model):
        if "db_prefix" in self.settings["backend"]:
            entity_prefix_used_in_db = self.settings["backend"]["db_prefix"]
        else:
            entity_prefix_used_in_db = ""

        if self.settings["backend"]["format"] == "sparql":
            database_interface = FreebaseInterface()
            expansion_strategy = OnlyFreebaseExpansionStrategy()
            facts = FreebaseFacts()
        elif self.settings["backend"]["format"] == "csv":
            database_interface = CsvInterface(self.settings["backend"]["csv_file"])
            expansion_strategy = AllThroughExpansionStrategy()
            facts = CsvFacts(self.settings["backend"]["csv_file"])

        database = HypergraphInterface(database_interface, expansion_strategy)
        candidate_generator = NeighborhoodCandidateGenerator(database, neighborhood_search_scope=1,
                                                             extra_literals=True)

        candidate_selector = TensorflowCandidateSelector(model, candidate_generator)
        candidate_selector.update_setting("facts", facts)
        return candidate_selector

    def search(self):
        configurations = []

        for k,v in self.settings["searchable"].items():
            options = [(k, option) for option in v.split(",")]
            configurations.append(options)

        for combination in itertools.product(*configurations):
            model = self.create_model()
            for k,v in combination:
                model.update_setting(k,v)
            model.initialize()

            parameter_string = ", ".join([k+"="+v for k,v in combination])
            yield parameter_string, model


    def retrieve_class_name(self, stack_name):
        if stack_name == "bow+dumb":
            return DumbEntityEmbeddingVsBagOfWords

        return None
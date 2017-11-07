from collections import defaultdict

from candidate_selection.tensorflow_models.model_parts.preprocessor import PreprocessorPart
from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder
from helpers.static import Static
from indexing.freebase_indexer import FreebaseIndexer
from indexing.glove_indexer import GloveIndexer
from indexing.lazy_indexer import LazyIndexer
import numpy as np


class AbstractTensorflowModel:

    is_tensorflow = True

    preprocessor = None
    components = None
    variables = None

    model_settings = None

    entity_scores = None

    """
    Methods for model initialization:
    """

    def __init__(self):
        self.model_settings = defaultdict(lambda : None)

    def initialize(self):
        self.components = []
        self.variables = TensorflowVariablesHolder()

        self.initialize_indexers()
        self.initialize_preprocessors()
        self.initialize_graph()

    def add_component(self, component):
        self.components.append(component)

    def prepare_tensorflow_variables(self, mode='train'):
        for component in self.components:
            component.prepare_tensorflow_variables(mode=mode)

    def initialize_preprocessors(self):
        preprocessor_stack_types = self.get_preprocessor_stack_types()
        preprocessor = PreprocessorPart(preprocessor_stack_types, self.word_indexer, self.entity_indexer,
                                        self.relation_indexer)
        preprocessor.initialize_all_preprocessors()
        self.preprocessor = preprocessor


    """
    Configuration of settings:
    """

    def update_setting(self, setting_string, value):
        if setting_string == "dimension":
            self.model_settings["entity_dimension"] = int(value)
            self.model_settings["word_dimension"] = int(value)
        elif setting_string in ["word_dimension", "entity_dimension", "n_lstms", "n_layers"]:
            self.model_settings[setting_string] = int(value)
        elif setting_string in ["static_entity_embeddings", "use_transformation"]:
            self.model_settings[setting_string] = True if value == "True" else False
        else:
            self.model_settings[setting_string] = value

    """
    Methods to construct the model:
    """

    def build_indexer(self, string, shape, default_embedding):
        if string is None or string == "none":
            return LazyIndexer(shape)
        elif string == "initialized" and default_embedding == "GloVe":
            key = default_embedding + "_" + str(shape[1])
            if key not in Static.embedding_indexers:
                Static.embedding_indexers[key] = GloveIndexer(shape[1])
            return Static.embedding_indexers[key]
        elif string == "initialized" and default_embedding == "Siva":
            key = default_embedding
            if key not in Static.embedding_indexers:
                Static.embedding_indexers[key] = FreebaseIndexer()
            return Static.embedding_indexers[key]

    """
    Methods to run the model:
    """

    def handle_variable_assignment(self, preprocessed_batch, mode="train"):
        for component in self.components:
            component.handle_variable_assignment(preprocessed_batch, mode)

        return self.variables.get_assignment_dict()

    """
    General tensorflow graph components:
    """

    def get_loss_graph(self, sum_examples=True):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_loss(self.entity_scores, sum_examples=sum_examples)

    def get_prediction_graph(self):
        if self.entity_scores is None:
            self.entity_scores = self.compute_entity_scores()

        return self.decoder.decode_to_prediction(self.entity_scores)


    """
    Heuristics:
    """


    def validate_example(self, example):
        candidates = example["neighborhood"].get_vertices(type="entities")
        target_vertices = example["gold_entities"]
        target_vertices_in_candidates = np.isin(target_vertices, candidates)

        return target_vertices_in_candidates.all()

    """
    Interface:
    """

    def get_preprocessor(self):
        return self.preprocessor

    def retrieve_entities(self, graph_index, entity_index):
        return self.preprocessor.retrieve_entities(graph_index, entity_index)

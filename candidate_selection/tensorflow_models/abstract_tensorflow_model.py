from collections import defaultdict

import numpy as np

from candidate_selection.tensorflow_variables_holder import TensorflowVariablesHolder


class AbstractTensorflowModel:

    is_tensorflow = True

    preprocessor = None
    components = None
    variables = None

    model_settings = None
    facts = None

    entity_scores = None

    graphs = None

    """
    Methods for model initialization:
    """

    def __init__(self, facts):
        self.model_settings = defaultdict(lambda : None)
        self.facts = facts

    def initialize(self):
        self.components = []
        self.variables = TensorflowVariablesHolder()
        self.graphs = {}

        #self.initialize_indexers()
        #self.initialize_preprocessors()
        self.initialize_graph()

    def add_component(self, component):
        self.components.append(component)

    def prepare_tensorflow_variables(self, mode='train'):
        for component in self.components:
            component.prepare_tensorflow_variables(mode=mode)

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor


    """
    Configuration of settings:
    """

    def update_setting(self, setting_string, value):
        if setting_string == "dimension":
            self.model_settings["entity_dimension"] = int(value)
            self.model_settings["word_dimension"] = int(value)
        elif setting_string in ["word_embedding_dimension", "entity_embedding_dimension", "relation_embedding_dimension", "n_lstms", "n_layers", "lstm_hidden_state_dimension", "nn_hidden_state_dimension", "gate_input_dim", "gate_input_layers"]:
            self.model_settings[setting_string] = int(value)
        elif setting_string in ["static_entity_embeddings", "use_transformation", "static_word_embeddings"]:
            self.model_settings[setting_string] = True if value == "True" else False
        elif setting_string in ["regularization_scale", "attention_dropout", "word_dropout", "transform_dropout", "edge_dropout"]:
            self.model_settings[setting_string] = float(value)
        else:
            self.model_settings[setting_string] = value

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

    def get_regularization(self):
        regularization = 0
        for component in self.components:
            regularization += component.get_regularization_term()
        return regularization

    def get_loss_graph(self, sum_examples=True, mode="train"):
        if mode not in self.graphs:
            self.graphs[mode] = self.compute_entity_scores(mode=mode)

        return self.decoder.decode_to_loss(self.graphs[mode], sum_examples=sum_examples) + self.get_regularization()

    def get_prediction_graph(self, mode="predict"):
        if mode not in self.graphs:
            self.graphs[mode] = self.compute_entity_scores(mode=mode)

        return self.decoder.decode_to_prediction(self.graphs[mode])


    """
    Heuristics:
    """

    def validate_example(self, example):
        candidates = example["neighborhood"].get_vertices(type="entities")
        target_vertices = example["gold_entities"]
        target_vertices_in_candidates = np.isin(target_vertices, candidates)

        return target_vertices_in_candidates.any()

    """
    Interface:
    """

    def get_preprocessor(self):
        return self.preprocessor

    def retrieve_entities(self, graph_index, entity_index):
        return self.preprocessor.retrieve_entities(graph_index, entity_index)

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
from candidate_selection.tensorflow_models.components.extras.target_comparator import TargetComparator
from candidate_selection.tensorflow_models.components.vector_encoders.multilayer_perceptron import MultilayerPerceptron
import tensorflow as tf


class NeuralNetworkOrFactorizationScorer(AbstractComponent):

    scoring_function_type = None
    use_transformation = None

    def __init__(self, model_settings, variables, variable_prefix=""):
        self.model_settings = model_settings

        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.variables = variables

        if self.model_settings["scoring_function"] == "neural_network":
            self.initialize_as_nn()
        elif self.model_settings["scoring_function"] == "factorization":
            self.initialize_as_factorization()

    def initialize_as_nn(self):
        self.scoring_function_type = "neural_network"
        self.final_transformation = MultilayerPerceptron([int(self.model_settings["lstm_hidden_state_dimension"] / 2) +
                                                          self.model_settings["entity_embedding_dimension"],
                                                          self.model_settings["nn_hidden_state_dimension"],
                                                          1],
                                                         self.variables,
                                                         variable_prefix=self.variable_prefix+"transformation",
                                                         l2_scale=self.model_settings["regularization_scale"],
                                                         dropout_rate=self.model_settings["transform_dropout"])
        self.target_comparator = TargetComparator(self.variables, variable_prefix=self.variable_prefix+"comparison_to_sentence", comparison="concat")
        self.sub_components = [self.target_comparator, self.final_transformation]

    def initialize_as_factorization(self):
        self.scoring_function_type = "factorization"
        self.use_transformation = self.model_settings["use_transformation"]
        self.target_comparator = TargetComparator(self.variables, variable_prefix=self.variable_prefix+"comparison_to_sentence", comparison="dot_product")
        self.sub_components = [self.target_comparator]

        if self.use_transformation:
            self.transformation = MultilayerPerceptron([int(self.model_settings["lstm_hidden_state_dimension"] / 2),
                                                        self.model_settings["entity_embedding_dimension"]],
                                                       self.variables,
                                                       variable_prefix=self.variable_prefix+"transformation",
                                                       l2_scale=self.model_settings["regularization_scale"],
                                                       dropout_rate=self.model_settings["transform_dropout"])

            self.sub_components += [self.transformation]

    def score(self, sentence_embeddings, entity_embeddings, mode="train"):
        if self.scoring_function_type == "neural_network":
            return self.score_nn(sentence_embeddings, entity_embeddings, mode=mode)
        else:
            return self.score_factorization(sentence_embeddings, entity_embeddings, mode=mode)

    def score_nn(self, sentence_embeddings, entity_embeddings, mode="train"):
        hidden = self.target_comparator.get_comparison_scores(sentence_embeddings, entity_embeddings)
        entity_scores = tf.squeeze(self.final_transformation.transform(hidden, mode=mode))

        return entity_scores

    def score_factorization(self, sentence_embeddings, entity_embeddings, mode="train"):
        if self.model_settings["use_transformation"]:
            sentence_embeddings = self.transformation.transform(sentence_embeddings, mode=mode)

        entity_scores = self.target_comparator.get_comparison_scores(sentence_embeddings,
                                                                     entity_embeddings)

        return entity_scores

    def prepare_tensorflow_variables(self, mode="train"):
        for component in self.sub_components:
            component.prepare_tensorflow_variables(mode=mode)


    def get_regularization_term(self):
        reg = 0
        for component in self.sub_components:
            reg += component.get_regularization_term()
        return reg

    def handle_variable_assignment(self, batch, mode):
        for component in self.sub_components:
            component.handle_variable_assignment(batch, mode)

import tensorflow as tf

from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent


class TargetComparator(AbstractComponent):

    variable_prefix = None
    variables = None
    comparison = None

    def __init__(self, variables, variable_prefix="", comparison="dot_product"):
        self.variables = variables
        self.variable_prefix = variable_prefix
        if self.variable_prefix != "":
            self.variable_prefix += "_"

        self.comparison = comparison

    def get_comparison_scores(self, all_target_embeddings, element_embeddings):
        all_target_embeddings = tf.Print(all_target_embeddings, [all_target_embeddings], summarize=25, message="Target embeddings: ")
        target_embeddings = tf.nn.embedding_lookup(all_target_embeddings, self.variables.get_variable(self.variable_prefix + "target_indices"))
        #target_embeddings = tf.Print(target_embeddings, [self.variables.get_variable(self.variable_prefix + "target_indices")], summarize=25)

        if self.comparison == "dot_product":
            comparison = tf.reduce_sum(target_embeddings * element_embeddings, axis=1)
        elif self.comparison == "sum":
            comparison = target_embeddings + element_embeddings
        elif self.comparison == "concat":
            comparison = tf.concat([target_embeddings, element_embeddings], axis=1)

        return comparison


    def prepare_tensorflow_variables(self, mode="train"):
        self.variables.add_variable(self.variable_prefix + "target_indices", tf.placeholder(tf.int32))

    def handle_variable_assignment(self, batch_dictionary, mode):
        self.variables.assign_variable(self.variable_prefix + "target_indices", batch_dictionary["neighborhood_input_model"].get_instance_indices())


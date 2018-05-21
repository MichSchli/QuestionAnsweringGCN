import numpy as np

from example_batcher.batch_function_objects.edge_feature_combiner import EdgeFeatureCombiner
from example_batcher.batch_function_objects.entity_normalization_handler import EntityNormalizationHandler
from example_batcher.batch_function_objects.entity_reshaping_utils import EntityReshapingUtils
from example_batcher.batch_function_objects.gold_label_handler import GoldLabelHandler
from example_batcher.batch_function_objects.sentence_feature_combiner import SentenceFeatureCombiner
from example_batcher.batch_function_objects.sentence_rehaping_utils import SentenceReshapingUtils
from example_batcher.batch_function_objects.vertex_feature_combiner import VertexFeatureCombiner
from example_batcher.batch_function_objects.vertex_index_combiner import VertexIndexCombiner


class Batch:

    examples = None
    vertex_index_combiner = None

    def __init__(self):
        self.examples = []
        self.vertex_index_combiner = VertexIndexCombiner(self)
        self.sentence_feature_combiner = SentenceFeatureCombiner(self)
        self.gold_label_handler = GoldLabelHandler(self)
        self.sentence_reshaping_utils = SentenceReshapingUtils(self)
        self.edge_feature_combiner = EdgeFeatureCombiner(self)
        self.entity_normalization_handler = EntityNormalizationHandler(self)
        self.entity_reshaping_utils = EntityReshapingUtils(self)
        self.vertex_feature_combiner = VertexFeatureCombiner(self)

    def get_examples(self):
        return self.examples

    """
    Counts:
    """

    def count_examples(self):
        return len(self.examples)

    def has_examples(self):
        return len(self.examples) > 0

    def count_all_entities(self):
        return sum(example.count_entities() for example in self.examples)

    def count_all_vertices(self):
        return sum(example.count_vertices() for example in self.examples)

    def get_entity_counts(self):
        return [example.count_entities() for example in self.examples]

    def get_sentence_lengths(self):
        return np.array([example.question.count_words() for example in self.examples], dtype=np.int32)

    """
    Index list getters:
    """

    def get_combined_mention_dummy_indices(self):
        return self.vertex_index_combiner.get_combined_mention_dummy_indices()

    def get_combined_word_vertex_indices(self):
        return self.vertex_index_combiner.get_combined_word_vertex_indices()

    def get_combined_entity_vertex_map_indexes(self):
        return self.vertex_index_combiner.get_combined_entity_vertex_map_indexes()

    def get_combined_sender_indices(self):
        return self.vertex_index_combiner.get_combined_sender_indices()

    def get_combined_receiver_indices(self):
        return self.vertex_index_combiner.get_combined_receiver_indices()

    def get_combined_sentence_vertex_indices(self):
        return self.vertex_index_combiner.get_combined_sentence_vertex_indices()

    """
    Sentence feature getters:
    """

    def get_padded_word_indices(self):
        return self.sentence_feature_combiner.get_padded_word_indices()

    def get_padded_pos_indices(self):
        return self.sentence_feature_combiner.get_padded_pos_indices()

    def get_padded_mention_indicators(self):
        return self.sentence_feature_combiner.get_padded_mention_indicators()

    def get_word_padding_mask(self):
        return self.sentence_feature_combiner.get_word_padding_mask()

    """
    Gold handlers:
    """

    def get_gold_vector(self):
        return self.gold_label_handler.get_gold_vector_use_only_entities()

    def get_gold_vector_use_all_vertices(self):
        return self.gold_label_handler.get_gold_vector_use_all_vertices()

    def get_padded_gold_matrix(self):
        return self.gold_label_handler.get_padded_gold_matrix()

    """
    Sentence reshaping:
    """

    def get_word_indexes_in_padded_sentence_matrix(self):
        return self.sentence_reshaping_utils.get_flat_word_indexes_arranged_in_padded_matrix()

    def get_word_indexes_in_flattened_sentence_matrix(self):
        return self.sentence_reshaping_utils.get_word_matrix_indexes_arranged_flat()

    """
    Entity reshaping:
    """

    def get_padded_entity_indexes(self):
        return self.entity_reshaping_utils.get_padded_entity_indexes()

    """
    Edge features:
    """

    def get_combined_edge_type_indices(self):
        return self.edge_feature_combiner.get_combined_edge_type_indices()

    def get_combined_gcn_type_edge_indices(self, i):
        return self.edge_feature_combiner.get_combined_gcn_type_edge_indices(i)

    def get_padded_edge_part_type_matrix(self):
        return self.edge_feature_combiner.get_padded_edge_part_type_matrix()

    """
    Vertex features:
    """

    def get_max_score_by_vertex(self):
        return self.vertex_feature_combiner.get_max_score_by_vertex()

    def get_combined_vertex_types(self):
        return self.vertex_feature_combiner.get_combined_vertex_types()

    """
    Entity normalization:
    """

    def get_normalization_by_vertex_count(self, weight_positives):
        return self.entity_normalization_handler.get_normalization_by_vertex_count(weight_positives=weight_positives)

    """
    Mention features:
    """

    def get_combined_mention_scores(self):
        index_lists = [np.copy([m.score for m in example.mentions]) for example in self.examples]

        return np.concatenate(index_lists).astype(np.float32)





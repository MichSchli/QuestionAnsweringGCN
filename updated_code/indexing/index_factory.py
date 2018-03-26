from indexing.indexes.entity_index import EntityIndex
from indexing.indexes.relation_index import RelationIndex
from indexing.indexes.word_index import WordIndex


class IndexFactory:

    def get(self, index_label, experiment_settings):
        if index_label == "relations":
            relation_index_type = experiment_settings["indexes"]["relation_index_type"]
            return RelationIndex(relation_index_type)
        elif index_label == "vertices":
            vertex_index_type = experiment_settings["indexes"]["vertex_index_type"]
            return EntityIndex(vertex_index_type)
        elif index_label == "words":
            word_index_type = experiment_settings["indexes"]["word_index_type"]
            return WordIndex(word_index_type)
        elif index_label == "pos":
            word_index_type = experiment_settings["indexes"]["pos_index_type"]
            return WordIndex(word_index_type)
from indexing.indexes.entity_index import EntityIndex
from indexing.indexes.relation_index import RelationIndex


class IndexFactory:

    def get(self, index_label, experiment_settings):
        if index_label == "relations":
            relation_index_type = experiment_settings["indexes"]["relation_index_type"]
            return RelationIndex(relation_index_type)
        elif index_label == "vertices":
            vertex_index_type = experiment_settings["indexes"]["vertex_index_type"]
            return EntityIndex(vertex_index_type)
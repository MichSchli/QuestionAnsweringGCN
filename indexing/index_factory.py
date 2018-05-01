from indexing.indexes.dep_embedding_index import DepIndex
from indexing.indexes.entity_index import EntityIndex
from indexing.indexes.glove_index import GloveIndex
from indexing.indexes.limited_freebase_relation_index import LimitedFreebaseRelationIndex
from indexing.indexes.pos_index import PosIndex
from indexing.indexes.relation_index import RelationIndex
from indexing.indexes.relation_part_index import RelationPartIndex
from indexing.indexes.word_index import WordIndex


class IndexFactory:

    cached_indexes = None

    def __init__(self):
        self.cached_indexes = {}

    def get(self, index_label, experiment_settings):
        index_choice, _, cache_label = self.read_index_settings(index_label, experiment_settings)

        if cache_label not in self.cached_indexes:
            self.cached_indexes[cache_label] = self.make_index(index_label, experiment_settings)

        return self.cached_indexes[cache_label]

    def read_index_settings(self, index_label, experiment_settings):
        setting_location = None
        if index_label == "pos":
            setting_location = "pos"
        elif index_label == "vertices":
            setting_location = "vertex"
        elif index_label == "words":
            setting_location = "word"
        elif index_label == "relations":
            setting_location = "relation"
        elif index_label == "relation_parts":
            setting_location = "relation_part"

        index_choice, dimension = experiment_settings["indexes"][setting_location + "_index_type"].split(":")

        cache_label = index_choice
        if index_choice == "glove":
            cache_label += "_" + dimension

        dimension = int(dimension)

        return index_choice, dimension, cache_label

    def make_index(self, index_label, experiment_settings):
        index_choice, dimension, _ = self.read_index_settings(index_label, experiment_settings)

        if index_label == "pos":
            return PosIndex(index_choice, dimension)
        elif index_label == "vertices":
            return EntityIndex(index_choice, dimension)
        elif index_label == "words" and index_choice == "dep":
            return DepIndex()
        elif index_label == "words" and index_choice == "glove":
            return GloveIndex(dimension)
        elif index_label == "words":
            return WordIndex(index_choice, dimension)
        elif index_label == "relations" and index_choice == "freebase_limited":
            return LimitedFreebaseRelationIndex(dimension)
        elif index_label == "relations":
            return RelationIndex(index_choice, dimension)
        elif index_label == "relation_parts":
            return RelationPartIndex(index_choice, dimension)
        else:
            print("ERROR: index \""+index_label+"\" not defined.")
            exit()
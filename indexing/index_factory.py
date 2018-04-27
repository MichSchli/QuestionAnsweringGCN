from indexing.indexes.dep_embedding_index import DepIndex
from indexing.indexes.entity_index import EntityIndex
from indexing.indexes.glove_index import GloveIndex
from indexing.indexes.pos_index import PosIndex
from indexing.indexes.relation_index import RelationIndex
from indexing.indexes.relation_part_index import RelationPartIndex
from indexing.indexes.word_index import WordIndex


class IndexFactory:

    cached_indexes = None

    def __init__(self):
        self.cached_indexes = {}

    def get(self, index_label, experiment_settings):
        if index_label == "relations":
            index_string = experiment_settings["indexes"]["relation_index_type"]
            if index_string not in self.cached_indexes:
                relation_index_type, dimension = index_string.split(":")
                dimension = int(dimension)
                self.cached_indexes[index_string] = RelationIndex(relation_index_type, dimension)

            return self.cached_indexes[index_string]

        elif index_label == "relation_parts":
            index_string = experiment_settings["indexes"]["relation_part_index_type"]
            if index_string not in self.cached_indexes:
                relation_index_type, dimension = index_string.split(":")
                dimension = int(dimension)
                self.cached_indexes[index_string] = RelationPartIndex(relation_index_type, dimension)

            return self.cached_indexes[index_string]

        elif index_label == "vertices":
            vertex_index_type, dimension = experiment_settings["indexes"]["vertex_index_type"].split(":")
            dimension = int(dimension)
            return EntityIndex(vertex_index_type, dimension)
        elif index_label == "words":
            word_index_type, dimension = experiment_settings["indexes"]["word_index_type"].split(":")
            dimension = int(dimension)

            if word_index_type.startswith("glove"):
                if "glove_"+str(dimension) not in self.cached_indexes:
                    self.cached_indexes["glove_" + str(dimension)] = GloveIndex(dimension)

                return self.cached_indexes["glove_"+str(dimension)]
            elif word_index_type == "dep":
                return DepIndex()
            else:
                return WordIndex(word_index_type, dimension)
        elif index_label == "pos":
            pos_index_type, dimension = experiment_settings["indexes"]["pos_index_type"].split(":")
            dimension = int(dimension)
            return PosIndex(pos_index_type, dimension)
        else:
            print("ERROR: index \""+index_label+"\" not defined.")
            exit()
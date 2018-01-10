from experiment_construction.index_construction.index_holder import IndexHolder
from experiment_construction.index_construction.indexes.freebase_indexer import FreebaseIndexer
from experiment_construction.index_construction.indexes.freebase_relation_indexer import FreebaseRelationIndexer

from experiment_construction.index_construction.indexes.glove_indexer import GloveIndexer
from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer


class IndexHolderFactory:

    indexes = None
    max_words = 40000
    max_entities = 40000
    max_relations = 6000

    # Hold a single index of each type in memory:
    index_cache = {"word": [None,None],
                   "entity": [None,None],
                   "relation": [None,None]}

    def __init__(self):
        self.indexes = {}

    def construct_indexes(self, settings):
        word_embedding_type = self.get_embedding_type("word_embedding_type", settings)
        entity_embedding_type = self.get_embedding_type("entity_embedding_type", settings)
        relation_embedding_type = self.get_embedding_type("relation_embedding_type", settings)

        word_embedding_dimension = self.get_embedding_width("word_embedding_dimension", settings)
        entity_embedding_dimension = self.get_embedding_width("entity_embedding_dimension", settings)
        relation_embedding_dimension = self.get_embedding_width("relation_embedding_dimension", settings)

        word_cache_string = word_embedding_type + "_" + str(word_embedding_dimension)
        if self.index_cache["word"][0] != word_cache_string:
            self.index_cache["word"][1] = self.build_indexer(word_embedding_type, [self.max_words, word_embedding_dimension])
        word_indexer = self.index_cache["word"][1]

        entity_cache_string = entity_embedding_type + "_" + str(entity_embedding_dimension)
        if self.index_cache["entity"][0] != entity_cache_string:
            self.index_cache["entity"][1] = self.build_indexer(entity_embedding_type,
                                                             [self.max_entities, entity_embedding_dimension])
        entity_indexer = self.index_cache["entity"][1]

        relation_cache_string = relation_embedding_type + "_" + str(relation_embedding_dimension)
        if self.index_cache["relation"][0] != relation_cache_string:
            self.index_cache["relation"][1] = self.build_indexer(relation_embedding_type,
                                                             [self.max_relations, relation_embedding_dimension])
        relation_indexer = self.index_cache["relation"][1]

        index = IndexHolder()
        index.word_indexer = word_indexer
        index.entity_indexer = entity_indexer
        index.relation_indexer = relation_indexer

        return index

    def get_embedding_width(self, embedding_string, settings):
        if "dimension" in settings["model"]:
            return int(settings["model"]["dimension"])
        elif embedding_string in settings["model"]:
            return int(settings["model"][embedding_string])
        else:
            return None

    def get_embedding_type(self, embedding_string, settings):
        if embedding_string in settings["model"]:
            embedding_type = settings["model"][embedding_string]
        else:
            embedding_type = "none"
        return embedding_type

    def build_indexer(self, type, shape):
        key = type + str(shape)
        if key in self.indexes:
            return self.indexes[key]
        elif type == "none":
            indexer = LazyIndexer(shape)
        elif type == "GloVe":
            indexer = GloveIndexer(shape[1])
        elif type == "Siva":
            indexer = FreebaseIndexer()
        elif type == "Relation":
            indexer = FreebaseRelationIndexer(shape, 10)

        self.indexes[key] = indexer
        return indexer

from helpers.logger import Logger
from indexing.freebase_indexer import FreebaseIndexer
from indexing.lazy_indexer import LazyIndexer


class Static:

    logger = None
    embedding_indexers = {}

    def get_freebase_entity_indexer(self):
        if self.embedding_indexers["Siva"] is None:
            self.embedding_indexers["Siva"] = FreebaseIndexer()
            return Static.embedding_indexers["Siva"]

    def get_freebase_relation_indexer(self):
        if self.embedding_indexers["Siva"] is None:
            self.embedding_indexers["Siva"] = FreebaseIndexer()
            return Static.embedding_indexers["Siva"]

    def get_toy_entity_indexer(self):
        return LazyIndexer((500,1000))

    def get_toy_relation_indexer(self):
        return LazyIndexer((500,100))
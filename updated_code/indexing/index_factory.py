from indexing.index import Index


class IndexFactory:

    def get(self, index_label):
        return Index()
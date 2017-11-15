from collections import defaultdict


class EdgeFilter:

    edge_counts = None
    relation_indexer = None

    def __init__(self, inner, edge_list_file, relation_indexer=None):
        self.inner = inner
        self.edge_counts = defaultdict(int)
        self.load_edge_list(edge_list_file)
        self.relation_indexer = None

    def load_edge_list(self, file):
        for line in open(file):
            parts = line.strip().split(' ')

            edge_name = parts[1]
            edge_count = int(parts[0])

            if self.relation_indexer is not None:
                edge_name = self.relation_indexer.index(edge_name)

            self.edge_counts[edge_name] = int(edge_count)

EdgeFilter(None, "/home/michael/Projects/QuestionAnswering/GCNQA/data/webquestions/edge_count.txt")
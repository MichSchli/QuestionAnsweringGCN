from collections import defaultdict
import numpy as np


class EdgeFilter:

    edge_counts = None
    edge_count_cutoff = None
    relation_indexer = None

    def __init__(self, inner, edge_list_file, edge_count_cutoff, relation_indexer=None):
        self.inner = inner
        self.edge_counts = defaultdict(int)
        self.relation_indexer = relation_indexer
        self.load_edge_list(edge_list_file)
        self.edge_count_cutoff = edge_count_cutoff

    def load_edge_list(self, file):
        for line in open(file):
            parts = line.strip().split('\t')

            edge_name = parts[0]
            edge_count = int(parts[0])

            if self.relation_indexer is not None:
                edge_name = self.relation_indexer.index_single_element(edge_name)

            self.edge_counts[edge_name] = int(edge_count)

    def enrich(self, instances):
        instances = self.inner.enrich(instances)

        for instance in instances:
            edges = instance["neighborhood"].entity_to_event_edges
            filtered_edges = self.filter_edges(edges)
            instance["neighborhood"].update_edges(filtered_edges, sources="entities", targets="events")

            edges = instance["neighborhood"].entity_to_entity_edges
            filtered_edges = self.filter_edges(edges)
            instance["neighborhood"].update_edges(filtered_edges, sources="entities", targets="entities")

            edges = instance["neighborhood"].event_to_entity_edges
            filtered_edges = self.filter_edges(edges)
            instance["neighborhood"].update_edges(filtered_edges, sources="events", targets="entities")

            instance["neighborhood"].update_vertices()

            yield instance

    def filter_edges(self, edges):
        counts = np.array([self.edge_counts[e] for e in edges[:, 1]])
        filtered_edges = edges[np.where(counts > self.edge_count_cutoff)]
        #print("Discarded " + str(counts.shape[0] - filtered_edges.shape[0]) + " of " + str(counts.shape[0]) + " edges.")
        return filtered_edges

from model.hypergraph_model import HypergraphModel


class IndexedInterface:

    inner_interface = None
    indexer = None

    def __init__(self, inner_interface, entity_indexer, event_indexer, relation_indexer):
        self.inner_interface = inner_interface
        self.entity_indexer = entity_indexer
        self.event_indexer = event_indexer
        self.relation_indexer = relation_indexer

    def get_neighborhood_hypergraph(self, vertices, hops=1, extra_literals=False):
        hypergraph = self.inner_interface.get_neighborhood_hypergraph(vertices, hops=hops, extra_literals=extra_literals)

        print(hypergraph.entity_to_entity_edges)
        event_indexes = {k:v for v, k in enumerate(hypergraph.event_vertices)}
        entity_indexes = {k:v for v, k in enumerate(hypergraph.entity_vertices)}

        hypergraph.entity_map = {k:v for k,v in enumerate(hypergraph.entity_vertices)}
        hypergraph.event_vertices = self.event_indexer.index(hypergraph.event_vertices)
        hypergraph.entity_vertices = self.entity_indexer.index(hypergraph.entity_vertices)

        hypergraph.event_to_entity_edges[:,1] = self.relation_indexer.index(hypergraph.event_to_entity_edges[:,1])
        hypergraph.entity_to_entity_edges[:,1] = self.relation_indexer.index(hypergraph.entity_to_entity_edges[:,1])
        hypergraph.entity_to_event_edges[:,1] = self.relation_indexer.index(hypergraph.entity_to_event_edges[:,1])

        for i, edge in enumerate(hypergraph.event_to_entity_edges):
            hypergraph.event_to_entity_edges[i][0] = event_indexes[edge[0]]
            hypergraph.event_to_entity_edges[i][2] = entity_indexes[edge[2]]

        for i, edge in enumerate(hypergraph.entity_to_event_edges):
            hypergraph.entity_to_event_edges[i][0] = entity_indexes[edge[0]]
            hypergraph.entity_to_event_edges[i][2] = event_indexes[edge[2]]

        for i, edge in enumerate(hypergraph.entity_to_entity_edges):
            hypergraph.entity_to_entity_edges[i][0] = entity_indexes[edge[0]]
            hypergraph.entity_to_entity_edges[i][2] = entity_indexes[edge[2]]

        print(hypergraph.entity_to_entity_edges)
        print(hypergraph.entity_vertices)

        exit()

        return hypergraph
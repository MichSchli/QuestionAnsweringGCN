class NewGraphReader:

    database_interface = None

    def __init__(self, database_interface):
        self.database_interface = database_interface

    def get_neighborhood_graph(self, entities):
        edge_query_result = self.database_interface.get_adjacent_edges(entities, target="entities", literals_only=False)
        edge_query_result_2 = self.database_interface.get_adjacent_edges(entities, target="events", literals_only=False)
        print(entities)
        print(edge_query_result_2.forward_edges)
        exit()
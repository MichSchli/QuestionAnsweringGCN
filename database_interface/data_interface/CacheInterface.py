class CacheInterface:

    inner = None

    def __init__(self, inner):
        self.inner = inner

    def get_adjacent_edges(self, node_identifiers, target="entities", literals_only=False):

        cache_strings = [identifier+"_"+target+"_"+literals_only for identifier in node_identifiers]

        #print("retrieving")
        edge_query_result = EdgeQueryResult()

        self.retrieve_edges_in_one_direction(node_identifiers, edge_query_result, subject=True, target=target, literals_only=literals_only)
        self.retrieve_edges_in_one_direction(node_identifiers, edge_query_result, subject=False, target=target, literals_only=literals_only)

        #print("done")
        return edge_query_result
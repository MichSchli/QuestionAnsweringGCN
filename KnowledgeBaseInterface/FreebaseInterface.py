from SPARQLWrapper import SPARQLWrapper, JSON

from KnowledgeBaseInterface.IKbInterface import IKbInterface


class FreebaseInterface(IKbInterface):
    endpoint = None
    prefix = None

    def __init__(self):
        self.endpoint = "http://localhost:8890/sparql"
        self.prefix = "http://rdf.freebase.com/ns/"

    def construct_neighbor_query(self, center_vertices, direction='s', limit=1000):
        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "FILTER (?" + direction + " in (" + ", ".join(center_vertices) + "))\n"
        query_string += "}\nLIMIT " + str(limit)

        return query_string

    def retrieve_one_neighborhood(self, node_identifiers, limit=None):
        forward = self.retrieve_one_neighborhood_graph(node_identifiers, limit=limit, subject=True)
        backward = self.retrieve_one_neighborhood_graph(node_identifiers, limit=limit, subject=False)



    def retrieve_one_neighborhood_graph(self, center_vertices, limit=1000, subject=True):
        sparql = SPARQLWrapper(self.endpoint)
        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "FILTER (" + "?s" if subject else "?o" + "in (" + ", ".join(center_vertices) + "))\n"
        query_string += "}\nLIMIT " + str(limit)

        query_string = self.construct_neighbor_query(center_vertices, limit=limit)

        print(query_string)

        sparql.setQuery(query_string)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        tuple_results = [None]* len(results["results"]["bindings"])
        for i,result in enumerate(results["results"]["bindings"]):
            tuple_results[i] = (result["s"]["value"], result["r"]["value"], result["o"]["value"])

        return tuple_results

if __name__ == "__main__":
    iface = FreebaseInterface()
    results = iface.retrieve_one_neighborhood_graph(["ns:m.014zcr", "ns:m.0q0b4"], limit=3000)
    for result in results:
        print(result)

from KnowledgeBaseInterface.IKbInterface import IKbInterface


class SparqlInterface(IKbInterface):
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

    def retrieve_one_neighborhood_graph(self, center_vertices, format="text", limit=1000, subject=True):
        sparql = SPARQLWrapper(self.endpoint)
        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "FILTER (?s in (" + ", ".join(center_vertices) + "))\n"
        query_string += "}\nLIMIT " + str(limit)

        query_string = self.construct_neighbor_query(center_vertices, limit=limit)

        print(query_string)

        sparql.setQuery(query_string)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        return results

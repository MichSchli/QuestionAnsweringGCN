from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from KnowledgeBaseInterface.IKbInterface import IKbInterface
import math


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

        forward_entities = forward[:,2]
        backward_entities = backward[:,0]

        new_entities = np.concatenate((forward_entities, backward_entities))
        new_entities = np.unique(new_entities)

        retrieved_edges = np.concatenate((forward, backward))

        return new_entities, retrieved_edges



    def retrieve_one_neighborhood_graph(self, center_vertices, limit=1000, subject=True, entities_per_query=10):
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setReturnFormat(JSON)

        number_of_batches = math.ceil(center_vertices.shape[0] / entities_per_query)

        result_chunks = [None]*number_of_batches
        for i,center_vertex_batch in enumerate(np.split(center_vertices, number_of_batches)):
            query_string = self.construct_neighbor_query(center_vertices, limit=limit, direction="s" if subject else "o")

            print(query_string)

            sparql.setQuery(query_string)
            results = sparql.query().convert()

            result_chunks[i] = [None]* len(results["results"]["bindings"])
            for j,result in enumerate(results["results"]["bindings"]):
                result_chunks[i][j] = [result["s"]["value"], result["r"]["value"], result["o"]["value"]]

        results = np.concatenate(result_chunks)
        print(results.shape)
        return results

if __name__ == "__main__":
    iface = FreebaseInterface()
    results = iface.retrieve_one_neighborhood(["ns:m.014zcr", "ns:m.0q0b4"], limit=3000)

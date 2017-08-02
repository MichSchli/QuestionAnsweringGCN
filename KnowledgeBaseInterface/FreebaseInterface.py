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
        query_string += "FILTER (?" + direction + " in (" + ", ".join(["ns:" + v.split("/ns/")[-1] for v in center_vertices]) + "))\n"
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

        #for edge in retrieved_edges:
        #    print(edge)

        return new_entities, retrieved_edges



    def retrieve_one_neighborhood_graph(self, center_vertices, limit=1000, subject=True, entities_per_query=100):
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setReturnFormat(JSON)

        number_of_batches = math.ceil(center_vertices.shape[0] / entities_per_query)
        print(center_vertices.shape[0])
        print(entities_per_query)
        print(number_of_batches)

        result_chunks = [None]*number_of_batches
        for i,center_vertex_batch in enumerate(np.array_split(center_vertices, number_of_batches)):
            query_string = self.construct_neighbor_query(center_vertex_batch, limit=limit, direction="s" if subject else "o")
            print(str(i) + " of " + str(number_of_batches))
            #print(query_string)

            sparql.setQuery(query_string)
            results = sparql.query().convert()

            result_chunks[i] = []
            for j,result in enumerate(results["results"]["bindings"]):
                if (subject and result["o"]["value"].startswith("http://rdf.freebase.com/ns/")) or (not subject and result["s"]["value"].startswith("http://rdf.freebase.com/ns/")): #("-" in result["o"]["value"] or "-" in result["s"]["value"] or "/key/" in result["r"]["value"] or "type.object.key" in result["r"]["value"] or "topic.description" in result["r"]["value"] or result["r"]["value"].endswith("label") or result["r"]["value"].endswith("name") or result["r"]["value"].endswith("alias") or result["r"]["value"].endswith("webpage")):
                    result_chunks[i].append([result["s"]["value"], result["r"]["value"], result["o"]["value"]])

        results = np.concatenate(result_chunks)
        print(results.shape)
        return results

if __name__ == "__main__":
    iface = FreebaseInterface()
    results = iface.retrieve_one_neighborhood(["ns:m.014zcr", "ns:m.0q0b4"], limit=3000)

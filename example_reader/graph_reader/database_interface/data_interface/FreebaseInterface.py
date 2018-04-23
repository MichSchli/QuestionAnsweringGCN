from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import math
import time

from example_reader.graph_reader.database_interface.data_interface.edge_query_result import EdgeQueryResult
from example_reader.graph_reader.database_interface.search_filters.prefix_filter import PrefixFilter


class FreebaseInterface:
    endpoint = None
    prefix = None
    max_entities_per_query = 200
    name_relation = None

    def __init__(self, ):
        self.endpoint = "http://localhost:8890/sparql"
        self.prefix = "http://rdf.freebase.com/ns/"
        self.name_relation = "http://www.w3.org/2000/01/rdf-schema#label"

        self.frontier_filter = PrefixFilter("http://rdf.freebase.com/ns/")


    """
    Construct a query to retrieve property fields associated to a set of vertices
    """
    def construct_property_query(self, center_vertices, forward=True):
        center = "s" if forward else "o"
        other = "o" if forward else "s"

        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "PREFIX rdf: <http://www.w3.org/2000/01/rdf-schema#>\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "values ?" + center + " {" + " ".join(
            ["ns:" + v.split("/ns/")[-1] for v in center_vertices]) + "}\n"
        query_string += "values ?r { rdf:label }\n"
        query_string += "FILTER (lang(?o) = \'en\')"
        query_string += "}"

        return query_string

    """
    Construct a query to retrieve all neighbors of a set of vertices.

     - hyperedges: If true, retrieve event neighbors. If false, retrieve entity neighbors.
     - forward: If true, retrieve edges where the centroids are the subject. If false, retrieve edges where the centroids are the object.
    """
    def construct_neighbor_query(self, center_vertices, hyperedges=True, forward=True):
        center = "s" if forward else "o"
        other = "o" if forward else "s"

        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "values ?" + center + " {" + " ".join(["ns:" + v.split("/ns/")[-1] for v in center_vertices]) + "}\n"
        query_string += "filter ( "

        if hyperedges:
            query_string += "( not exists { ?" + other + " ns:type.object.name ?name } && !isLiteral(?" + other + ") && strstarts(str(?"+other+"), \"" + self.prefix + "\") )"
        else:
            query_string += "( exists { ?" + other + " ns:type.object.name ?name } || isLiteral(?" + other + ") )"

        query_string += "\n&& (!isLiteral(?" + other + ") || lang(?" + other + ") = 'en' || datatype(?" + other + ") != xsd:string || datatype(?" + other + ") != rdf:langString )"
        # Take out all schemastaging for now. Might consider putting some parts back in later:
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/base.schemastaging\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/key/wikipedia\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/common.topic.webpage\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/type.object.key\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/base.yupgrade.user.topics\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/common.topic.description\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/common.document.text\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/type.type.instance\" )"
        query_string += "\n&& !strstarts(str(?r), \"http://rdf.freebase.com/ns/type.object.type\" )"
        query_string += " )\n"

        query_string += "}"


        return query_string

    """
    Construct a query to retrieve all neighbors of a set of vertices
    """
    def construct_neighbor_query_old(self, center_vertices, direction='s'):
        property_list = ["ns:type.object.name"]
        opposite_direction = "o" if direction == "s" else "s"

        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"

        query_string += "\n".join(["?" + opposite_direction + " " + prop + " ?prop" for prop in property_list])

        query_string += "FILTER (?" + direction + " in (" + ", ".join(["ns:" + v.split("/ns/")[-1] for v in center_vertices]) + "))\n"
        query_string += "}"

        return query_string

    """
    Retrieve the 1-neighborhood of a set of vertices in the hypergraph
    """
    def get_adjacent_edges(self, node_identifiers, target="entities", literals_only=False):
        edge_query_result = EdgeQueryResult()

        self.retrieve_edges_in_one_direction(node_identifiers, edge_query_result, subject=True, target=target, literals_only=literals_only)
        self.retrieve_edges_in_one_direction(node_identifiers, edge_query_result, subject=False, target=target, literals_only=literals_only)

        #print("done")
        return edge_query_result


    """
    Retrieve names and append the property to the hypergraph
    """
    def retrieve_and_append_name(self, hypergraph, ingoing_edges, outgoing_edges):
        new_vertices = self.retrieve_new_vertices(ingoing_edges, outgoing_edges)
        names = self.get_properties(new_vertices, "ns:type.object.name")
        hypergraph.set_vertex_properties(names, "name")

    """
    Retrieve all new, unique subject/objects
    """
    def retrieve_new_vertices(self, ingoing_edges, outgoing_edges):
        outgoing_vertices = self.slice_empty(outgoing_edges, 2)
        ingoing_vertices = self.slice_empty(ingoing_edges, 0)
        new_vertices = np.concatenate((outgoing_vertices, ingoing_vertices))
        new_vertices = np.unique(new_vertices)
        return new_vertices

    def slice_empty(self, outgoing_edges, slice):
        if outgoing_edges.shape[0] > 0:
            outgoing_vertices = outgoing_edges[:, slice]
        else:
            outgoing_vertices = np.array([])
        return outgoing_vertices

    """
    Retrieve properties from DB
    """
    def get_property(self, vertices, property):
        db_interface = self.initialize_sparql_interface()
        number_of_batches = math.ceil(vertices.shape[0] / self.max_entities_per_query)

        result_list = []
        for i,center_vertex_batch in enumerate(np.array_split(vertices, number_of_batches)):
            query_string = self.construct_property_query(center_vertex_batch, property)

            results = self.execute_query(db_interface, query_string)

            for j,result in enumerate(results["results"]["bindings"]):
                result_list.append([
                    result["s"]["value"],
                    result["prop"]["value"]]
                )

        result_list = np.array(result_list)
        return result_list


    """
    Retrieve edges from DB going one direction.
    """
    def retrieve_edges_in_one_direction(self, center_vertices, edge_query_result, subject=True, target="entities", literals_only=False):
        db_interface = self.initialize_sparql_interface()

        number_of_batches = math.ceil(center_vertices.shape[0] / self.max_entities_per_query)

        for i,center_vertex_batch in enumerate(np.array_split(center_vertices, number_of_batches)):
            db_interface = self.initialize_sparql_interface()

            if target == "entities":
                if not literals_only:
                    query_string = self.construct_neighbor_query(center_vertex_batch, hyperedges=False, forward=subject)
                else:
                    query_string = self.construct_property_query(center_vertex_batch, forward=subject)
            else:
                query_string = self.construct_neighbor_query(center_vertex_batch, hyperedges=True, forward=subject)
            #print("#", end='', flush=True)

            results = self.execute_query(db_interface, query_string)

            if results is None:
                print("Query failed to work five times. Skipping.")
                continue

            #print(query_string)
            #print(len(results["results"]["bindings"]))

            for j,result in enumerate(results["results"]["bindings"]):
                #print(result["s"]["value"])
                #print(result["r"]["value"])
                #print(result["o"]["value"])
                # Retrieving literals only crashes SPARQL DB. So, we filter in python instead:
                #if literals_only:
                #    print(result)
                if literals_only and subject and result["o"]["type"] != "literal":
                    continue
                elif literals_only and not subject and result["s"]["type"] != "literal":
                    continue

                if target == "event" and subject and not (len(result["o"]["value"]) > 28 and result["o"]["value"][28] == ""):
                    continue
                elif target == "event" and object and not (len(result["s"]["value"]) > 28 and result["s"]["value"][28] == ""):
                    continue

                if result["r"]["value"] == self.name_relation:
                    edge_query_result.append_name(result["s"]["value"], result["o"]["value"])
                else:
                    edge_query_result.append_edge([
                        result["s"]["value"],
                        result["r"]["value"],
                        result["o"]["value"]], forward=subject
                    )
                    if subject:
                        edge_query_result.append_vertex(result["o"]["value"],result["o"]["type"])
                    else:
                        edge_query_result.append_vertex(result["s"]["value"],result["s"]["type"])

            #exit()
        #print("\r" + (i+1) * " "+"\r", end="", flush=True)

    def execute_query(self, db_interface, query_string):
        #print(query_string)
        db_interface.setQuery(query_string)
        retrieved = False
        trial_counter = 0
        while not retrieved:
            try:
                results = db_interface.query().convert()
                retrieved = True
            except:
                trial_counter += 1
                if trial_counter == 5:
                    return None

                print("Query failed. Reattempting in 5 seconds...\n")
                print(query_string)

                time.sleep(5)
        return results

    def initialize_sparql_interface(self):
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setReturnFormat(JSON)
        return sparql

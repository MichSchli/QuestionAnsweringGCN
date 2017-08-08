from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from database_interface.IKbInterface import IKbInterface
import math
import time

from database_interface.data_interface.edge_query_result import EdgeQueryResult
from database_interface.search_filters.prefix_filter import PrefixFilter
from model.hypergraph import Hypergraph


class FreebaseInterface(IKbInterface):
    endpoint = None
    prefix = None
    max_entities_per_query = 100

    def __init__(self):
        self.endpoint = "http://localhost:8890/sparql"
        self.prefix = "http://rdf.freebase.com/ns/"

        self.frontier_filter = PrefixFilter("http://rdf.freebase.com/ns/")

    """
    Construct a query to retrieve property fields associated to a set of vertices
    """
    def construct_property_query(self, vertices, property):
        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s " + property + " ?prop .\n"
        query_string += "FILTER (?s in (" + ", ".join(["ns:" + v.split("/ns/")[-1] for v in vertices]) + "))\n"
        query_string += "}\n"

        return query_string

    """
    Construct a query to retrieve all neighbors of a set of vertices
    """
    def construct_neighbor_query(self, center_vertices, direction='s', limit=1000):
        query_string = "PREFIX ns: <" + self.prefix + ">\n"
        query_string += "select * where {\n"
        query_string += "?s ?r ?o .\n"
        query_string += "FILTER (?" + direction + " in (" + ", ".join(["ns:" + v.split("/ns/")[-1] for v in center_vertices]) + "))\n"
        query_string += "}\nLIMIT " + str(limit)

        return query_string

    """
    Retrieve the 1-neighborhood of a set of vertices in the hypergraph
    """
    def retrieve_one_neighborhood(self, node_identifiers, limit=None):
        edge_query_result = EdgeQueryResult()

        self.retrieve_one_neighborhood_graph(node_identifiers, edge_query_result, subject=True)
        self.retrieve_one_neighborhood_graph(node_identifiers, edge_query_result, subject=False)

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
    def get_properties(self, vertices, property):
        db_interface = self.initialize_sparql_interface()
        number_of_batches = math.ceil(vertices.shape[0] / self.max_entities_per_query)

        result_list = []
        for i,center_vertex_batch in enumerate(np.array_split(vertices, number_of_batches)):
            query_string = self.construct_property_query(center_vertex_batch, property)
            print("#", end='', flush=True)

            results = self.execute_query(db_interface, query_string)

            for j,result in enumerate(results["results"]["bindings"]):
                result_list.append([
                    result["s"]["value"],
                    result["prop"]["value"]]
                )

        result_list = np.array(result_list)
        print("\r" + (i+1) * " "+"\r", end="", flush=True)
        return result_list

    """
    Retrieve edges from DB going one direction.
    """
    def retrieve_edges_in_one_direction(self, center_vertices, edge_query_result, subject=True):
        db_interface = self.initialize_sparql_interface()

        number_of_batches = math.ceil(center_vertices.shape[0] / self.max_entities_per_query)

        for i,center_vertex_batch in enumerate(np.array_split(center_vertices, number_of_batches)):
            query_string = self.construct_neighbor_query(center_vertex_batch, limit=limit, direction="s" if subject else "o")
            print("#", end='', flush=True)

            results = self.execute_query(db_interface, query_string)

            for j,result in enumerate(results["results"]["bindings"]):
                edge_query_result.append_edge([
                    result["s"]["value"],
                    result["r"]["value"],
                    result["o"]["value"]]
                )
                edge_query_result.append_vertex(result["s"]["value"],result["s"]["type"])
                edge_query_result.append_vertex(result["o"]["value"],result["o"]["type"])

        print("\r" + (i+1) * " "+"\r", end="", flush=True)

    def execute_query(self, db_interface, query_string):
        db_interface.setQuery(query_string)
        retrieved = False
        while not retrieved:
            try:
                results = db_interface.query().convert()
                retrieved = True
            except:
                print("Query failed. Reattempting in 5 seconds...")
                time.sleep(5)
        return results

    def initialize_sparql_interface(self):
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setReturnFormat(JSON)
        return sparql
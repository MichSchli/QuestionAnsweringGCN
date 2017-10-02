from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
from preprocessing.read_conll_files import ConllReader
import itertools

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

sparql = SPARQLWrapper("http://localhost:8890/sparql")

gold_reader = ConllReader(output="gold")
sentence_reader = ConllReader(output="entities", entity_prefix="ns:")

def generate_1_query(centroids, golds, forward_edges=True):
    centroid_symbol = "s" if forward_edges else "o"
    gold_symbol = "o" if forward_edges else "s"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect * where {"
    query += "\n\t?s ?r ?o ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join(["\""+g+"\"" for g in golds]) + " }"
    query += "\n}"

    return query

def get_1_paths(centroids, golds):
    query = generate_1_query(centroids, golds)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for r in results["results"]["bindings"]:
        yield r["r"]

    query = generate_1_query(centroids, golds, forward_edges=False)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for r in results["results"]["bindings"]:
        yield r["r"]

def generate_2_query(centroids, golds, forward_1_edges=True, forward_2_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r1 ?i ." if forward_1_edges else "?i ?r1 ?s"
    second_edge_string = "?i ?r2 ?o ." if forward_2_edges else "?o ?r2 ?i"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect * where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + second_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join(["\""+g+"\"" for g in golds]) + " }"
    query += "\n}"

    return query

def get_2_paths(centroids, golds):
    yield from get_2_paths_internal(centroids, golds, True, True)
    yield from get_2_paths_internal(centroids, golds, True, False)
    yield from get_2_paths_internal(centroids, golds, False, True)
    yield from get_2_paths_internal(centroids, golds, False, False)


def get_2_paths_internal(centroids, golds, forward_1, forward_2):
    query = generate_2_query(centroids, golds, forward_1, forward_2)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        yield r["r1"], r["r2"]


counter = 0
for gold, sentence in zip(gold_reader.parse_file(args.file), sentence_reader.parse_file(args.file)):
    print(counter)
    counter += 1
    for edge in get_1_paths(sentence, gold):
        print(edge)

    for edge_1,edge_2 in get_2_paths(sentence, gold):
        print(edge_1 + " " + edge_2)

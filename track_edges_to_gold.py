from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
from preprocessing.read_conll_files import ConllReader

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
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join(golds) + " }"
    query += "\n}"

    return query


for gold, sentence in zip(gold_reader.parse_file(args.file), sentence_reader.parse_file(args.file)):
    query = generate_1_query(sentence, gold)
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    print(results)
    exit()

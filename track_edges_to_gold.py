from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
from preprocessing.read_conll_files import ConllReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

sparql = SPARQLWrapper("http://localhost:8890/sparql")

gold_reader = ConllReader(output="gold")
sentence_reader = ConllReader(output="entities", entity_prefix="ns:")

def generate_1_query(centroids, golds):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect count(*) where {"
    query += "\n?s ?r ?o ."
    query += "\nvalues ?s { " + " ".join(centroids) + " }"


for gold, sentence in zip(gold_reader.parse_file(args.file)[:3], sentence_reader.parse_file(args.file)[:3]):
    query = generate_1_query(sentence, gold)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    print(results)

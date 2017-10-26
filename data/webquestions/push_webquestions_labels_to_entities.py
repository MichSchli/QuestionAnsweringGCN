import argparse

from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper

from helpers.read_conll_files import ConllReader

parser = argparse.ArgumentParser(description='Parses webquestions files with entity annotations to our internal format.')
parser.add_argument('--input_file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

def strip_prefix(string):
    return string[len("http://rdf.freebase.com/ns/"):]

def retrieve_entity(string):
    sparql = SPARQLWrapper("http://localhost:8890/sparql")
    sparql.setReturnFormat(JSON)

    query_string = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    query_string += "select * where {\n"
    query_string += "?s ?r ?o .\n"
    query_string += "values ?s { " + string + "@en }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "}"

    sparql.setQuery(query_string)
    results = sparql.query().convert()
    return results

newline_counter = 0
for line in open(args.input_file):
    line = line.strip()

    if not line:
        print(line)
        newline_counter += 1
        continue

    if newline_counter % 3 == 2:
        literal = line.split('\t')[-1]
        entity = retrieve_entity(literal)
        entity = strip_prefix(entity)
        print(entity)
        exit()
    else:
        print(line)

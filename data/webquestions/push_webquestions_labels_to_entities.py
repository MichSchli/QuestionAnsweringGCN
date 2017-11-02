import argparse
import numpy as np
import time
from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper

parser = argparse.ArgumentParser(description='Parses webquestions files with entity annotations to our internal format.')
parser.add_argument('--input_file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

def strip_prefix(string):
    return string[len("http://rdf.freebase.com/ns/"):]

def retrieve_entity(centroids, string):
    sparql = SPARQLWrapper("http://localhost:8890/sparql")
    sparql.setReturnFormat(JSON)

    query_string = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    query_string += "select ?y where {\n"

    query_string += "{\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?y { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?s ?r1 ?y .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?y ?r1 ?s .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?s ?r1 ?e .\n"
    query_string += "?e ?r2 ?y .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "filter ( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?e ?r1 ?s .\n"
    query_string += "?e ?r2 ?y .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "filter ( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?s ?r1 ?e .\n"
    query_string += "?y ?r2 ?e .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "filter ( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query_string += "}\n"
    query_string += "UNION\n"

    query_string += "{\n"
    query_string += "?e ?r1 ?s .\n"
    query_string += "?y ?r2 ?e .\n"
    query_string += "?y ?r ?o .\n"
    query_string += "values ?s { " + " ".join(["ns:" + v for v in centroids]) + " }\n"
    query_string += "values ?r { ns:type.object.name }\n"
    query_string += "values ?o { \"" + string + "\"@en }\n"
    query_string += "filter ( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query_string += "}\n"

    query_string += "}"

    sparql.setQuery(query_string)


    counter = 0
    while counter < 5:
        counter += 1
        try:
            results = sparql.query().convert()
            return np.unique([r['y']['value'] for r in results['results']['bindings']])
        except:
            print("Query failed. Reattempting in 5 seconds...")
            time.sleep(5)

    return np.array([])

#return np.unique([r['y']['value'] for r in results['results']['bindings']])

newline_counter = 0
centroids = []
shitty_counter = 0

zero_count = 0
too_many_count = 0

for line in open(args.input_file):
    line = line.strip()

    if not line:
        print(line)
        newline_counter += 1
        continue

    if newline_counter % 3 == 0:
        centroids = []

    if newline_counter % 3 == 1:
        centroids.append(line.split("\t")[2])

    if newline_counter % 3 == 2:
        literal = line.split('\t')[-1]
        entity = retrieve_entity(centroids, literal)
        #entity = strip_prefix(entity)
        shitty_counter += 1
        if entity.shape[0] != 1:
            print("=====")
            print(str(shitty_counter) + "\t" + str(entity))
            print("=====")
            time.sleep(5)
        else:
            print(str(shitty_counter) + "\t" + str(entity))

        if entity.shape[0] > 1:
            too_many_count += 1
        elif entity.shape[0] == 0:
            zero_count += 1
    else:
        print(line)

print(too_many_count)
print(zero_count)

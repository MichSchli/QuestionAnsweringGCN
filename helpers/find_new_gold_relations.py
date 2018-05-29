import argparse
from dateutil.parser import parse as date_parse
import json

import time

from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

dep_dict = {}

sparql = SPARQLWrapper("http://localhost:8890/sparql")
sparql.setReturnFormat(JSON)

# This is some serious bullshit:
def format_string_for_freebase(s):
    try:
        float(s)
        return s
    except ValueError:
        pass

    try:
        date_parse(s)
        return s
    except ValueError:
        return "\""+s+"\"@en"

def generate_1_query(centroids, golds, forward_edges=True):
    centroid_symbol = "s" if forward_edges else "o"
    gold_symbol = "o" if forward_edges else "s"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect ?r where {"
    query += "\n\t?s ?r ?o ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"
    query += "\n}"

    return query

def generate_1_query_with_name(centroids, golds, forward_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r ?i" if forward_edges else "?i ?r ?s"
    second_edge_string = "?i ns:type.object.name ?o"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect ?r where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + second_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"
    query += "\n}"

    return query

def get_1_paths(centroids, golds):
    query = generate_1_query(centroids, golds)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"]

    query = generate_1_query(centroids, golds, forward_edges=False)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"]

    query = generate_1_query_with_name(centroids, golds)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"]

    query = generate_1_query_with_name(centroids, golds, forward_edges=False)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"]

def generate_2_query(centroids, golds, forward_1_edges=True, forward_2_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r1 ?i" if forward_1_edges else "?i ?r1 ?s"
    second_edge_string = "?i ?r2 ?o" if forward_2_edges else "?o ?r2 ?i"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect ?r1 ?r2 where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + second_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"
    query += "\n}"

    return query

def generate_2_query_through_event(centroids, golds, forward_1_edges=True, forward_2_edges=True, forward_3_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r1 ?e" if forward_1_edges else "?e ?r1 ?s"
    third_edge_string = "?e ?r3 ?o" if forward_3_edges else "?o ?r3 ?e"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect ?r1 ?r2 ?r3 where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + third_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"

    query += "filter ( "
    query += "( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/base.schemastaging\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/key/wikipedia\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.webpage\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/type.object.key\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/base.yupgrade.user.topics\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.description\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/base.schemastaging\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/key/wikipedia\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.webpage\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/type.object.key\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/base.yupgrade.user.topics\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.description\" )"
    query += "\n\t)"

    query += "\n}"

    return query

def generate_2_query_through_event_with_name(centroids, golds, forward_1_edges=True, forward_2_edges=True, forward_3_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r1 ?e" if forward_1_edges else "?e ?r1 ?s"
    second_edge_string = "?e ?r2 ?i" if forward_2_edges else "?i ?r2 ?e"
    third_edge_string = "?i ns:type.object.name ?o"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect ?r1 ?r2 ?r3 where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + second_edge_string + " ."
    query += "\n\t" + third_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"

    query += "filter ( "
    query += "( not exists { ?e ns:type.object.name ?name } && !isLiteral(?e) && strstarts(str(?e), \"http://rdf.freebase.com/ns/\") )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/base.schemastaging\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/key/wikipedia\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.webpage\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/type.object.key\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/base.yupgrade.user.topics\" )"
    query += "\n\t&& !strstarts(str(?r1), \"http://rdf.freebase.com/ns/common.topic.description\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/base.schemastaging\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/key/wikipedia\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.topic_equivalent_webpage\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.webpage\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/type.object.key\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/base.yupgrade.user.topics\" )"
    query += "\n\t&& !strstarts(str(?r2), \"http://rdf.freebase.com/ns/common.topic.description\" )"
    query += "\n\t)"

    query += "\n}"

    return query

def get_2_paths(centroids, golds):
    yield from get_2_paths_internal(centroids, golds, True, True)
    yield from [(x[0], x[1]+".inverse") for x in get_2_paths_internal(centroids, golds, True, False)]
    yield from get_2_paths_internal(centroids, golds, False, True)
    yield from [(x[0]+".inverse", x[1]+".inverse") for x in get_2_paths_internal(centroids, golds, False, False)]


def get_2_paths_internal(centroids, golds, forward_1, forward_2):
    query = generate_2_query_through_event(centroids, golds, forward_1, forward_2)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r1"]["value"], r["r2"]["value"]

    query = generate_2_query_through_event_with_name(centroids, golds, forward_1, forward_2)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r1"]["value"], r["r2"]["value"]


def get_best_relation_pair(entity, golds):
    one_relations = list(get_1_paths(["ns:"+entity], golds))
    two_relations = list(get_2_paths(["ns:"+entity], golds))

    print(one_relations)
    print(two_relations)

def execute_query(db_interface, query_string):
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

with open(args.input_file) as data_file:
    reading_sentence = True
    reading_entities = False
    sentence = []
    for line in data_file:
        line = line.strip()

        if line and reading_sentence:
            parts = line.split("\t")
            if not parts[1]:
                parts[1] = "<NaN>"
            sentence.append(parts[1])
        elif not line and reading_sentence:

            print("")
            print("SENTENCE: " + " ".join(sentence) + "\n")

            reading_sentence = False
            reading_entities = True
            sentence = []
            entities = []
            golds = []
        elif line and reading_entities:
            entities.append(line.split("\t")[2])
        elif line and not reading_sentence and not reading_entities:
            golds.append(line.split("\t")[1])
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

            for entity in entities:
                best_relation_pair = get_best_relation_pair(entity, golds)
                print(entity)
                print(best_relation_pair)

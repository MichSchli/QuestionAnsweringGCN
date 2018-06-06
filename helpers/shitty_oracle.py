import argparse
from dateutil.parser import parse as date_parse
import json

import time
import numpy as np

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
        yield r["r"]["value"] + ".inverse"

    query = generate_1_query_with_name(centroids, golds)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"]

    query = generate_1_query_with_name(centroids, golds, forward_edges=False)
    results = execute_query(sparql, query)

    for r in results["results"]["bindings"]:
        yield r["r"]["value"] + ".inverse"

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
    yield from [(x[0]+".inverse", x[1]) for x in get_2_paths_internal(centroids, golds, False, True)]
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

def get_1_entities(centroids, target_edge, forward):
    result_list = []

    if len(centroids) == 0:
        return result_list

    db_interface = sparql

    center = "s" if forward else "o"
    other = "o" if forward else "s"

    query_string = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    query_string += "select * where {\n"
    query_string += "?s " + "ns:" + target_edge.split("/ns/")[-1] + " ?o .\n"
    query_string += "values ?" + center + " {" + " ".join(["ns:" + v.split("/ns/")[-1] for v in centroids]) + "}\n"
    query_string += "}"

    results = execute_query(db_interface, query_string)
    result_list.extend([r[other]["value"] for r in results["results"]["bindings"]])

    return result_list

def get_2_entities(centroids, target_edge, forward, target_edge_2, forward_2):
    result_list = []

    if len(centroids) == 0:
        return result_list

    db_interface = sparql

    edge_1 = "?s ns:" + target_edge.split("/ns/")[-1] + " ?e ." if forward else "?e ns:" + target_edge.split("/ns/")[-1] + " ?s ."
    edge_2 = "?e ns:" + target_edge_2.split("/ns/")[-1] + " ?o ." if forward_2 else "?o ns:" + target_edge_2.split("/ns/")[-1] + " ?e ."

    query_string = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    query_string += "select * where {\n"
    query_string += edge_1
    query_string += edge_2
    query_string += "values ?s {" + " ".join(["ns:" + v.split("/ns/")[-1] for v in centroids]) + "}\n"
    query_string += "}"

    results = execute_query(db_interface, query_string)
    result_list.extend([r["o"]["value"] for r in results["results"]["bindings"]])

    return result_list

def get_name(entity):
    if not entity.startswith("http://rdf.freebase.com/ns/"):
        return []

    db_interface = sparql

    query_string = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    query_string += "select * where {\n"
    query_string += "ns:" + entity.split("/ns/")[-1] + " ns:type.object.name ?o .\n"
    query_string += "filter ( "
    query_string += "\nlang(?o) = 'en'"
    query_string += " )\n"

    query_string += "}"
    results = execute_query(db_interface, query_string)
    return [r["o"]["value"] for r in results["results"]["bindings"]]


def get_f1(full_predictions, true_labels):
    true_positives = np.isin(full_predictions, true_labels)
    false_positives = np.logical_not(true_positives)
    false_negatives = np.isin(true_labels, full_predictions, invert=True)

    if np.sum(true_positives) + np.sum(false_positives) == 0:
        inner_precision = 1.0
    else:
        inner_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

    if np.sum(true_positives) + np.sum(false_negatives) == 0:
        inner_recall = 1.0
    else:
        inner_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    if inner_precision + inner_recall > 0:
        inner_f1 = 2 * (inner_precision * inner_recall) / (inner_precision + inner_recall)
    else:
        inner_f1 = 0

    return inner_precision, inner_recall, inner_f1

def get_one_relation_prediction(entity, relation, golds, forward=False):
    relation = "http://rdf.freebase.com/ns/" + relation

    retrieved = get_1_entities([entity], relation, forward)
    names = [get_name(r) for r in retrieved]
    full_predictions = [n if len(n) > 0 else [r] for n, r in zip(names, retrieved)]
    full_predictions = np.concatenate(full_predictions) if len(full_predictions) > 0 else full_predictions
    full_predictions = np.unique(full_predictions)

    return get_f1(full_predictions, golds)

def get_two_relation_prediction(entity, relation_1, relation_2, golds):
    if relation_1.endswith(".inverse"):
        actual_relation_1 = relation_1[:-8]
        forward_1 = False
    else:
        actual_relation_1 = relation_1
        forward_1 = True

    if relation_2.endswith(".inverse"):
        actual_relation_2 = relation_2[:-8]
        forward_2 = False
    else:
        actual_relation_2 = relation_2
        forward_2 = True

    actual_relation_1 = "http://rdf.freebase.com/ns/" + actual_relation_1
    actual_relation_2 = "http://rdf.freebase.com/ns/" + actual_relation_2

    retrieved = get_2_entities([entity], actual_relation_1, forward_1, actual_relation_2, forward_2)
    names = [get_name(r) for r in retrieved]
    full_predictions = [n if len(n) > 0 else [r] for n, r in zip(names, retrieved)]
    full_predictions = np.concatenate(full_predictions) if len(full_predictions) > 0 else full_predictions
    full_predictions = np.unique(full_predictions)
    return get_f1(full_predictions, golds)

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
                return {"results":{"bindings": []}}

            #print("Query failed. Reattempting in 5 seconds...\n")
            #print(query_string)

            time.sleep(5)
    return results

average_precision = 0
average_recall = 0
average_f1 = 0
count = 0

with open(args.input_file) as data_file:

    count = 0
    average_f1 = 0
    average_precision = 0
    average_recall = 0
    for line in data_file:
        line = line.strip()

        if line:
            parts = line.split("\t")
            one_relation = parts[1].endswith(".1") or parts[2].endswith(".1")

            if one_relation:
                forward = parts[1].endswith(".1")
                p, r, f1 = get_one_relation_prediction(parts[0], parts[1][:-2], parts[3].split(","), forward=forward)
            else:
                p, r, f1 = get_two_relation_prediction(parts[0], parts[1], parts[2], parts[3].split(","))
                print(f1)
                exit()


print(average_precision / count)
print(average_recall / count)
print(average_f1 / count)

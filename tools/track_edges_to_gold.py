from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
from preprocessing.read_conll_files import ConllReader
import itertools
from dateutil.parser import parse as date_parse

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
    query += "\n\nselect ?r where {"
    query += "\n\t?s ?r ?o ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"
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
    second_edge_string = "?e ?r2 ?i" if forward_2_edges else "?i ?r2 ?e"
    third_edge_string = "?i ?r3 ?o" if forward_3_edges else "?o ?r3 ?i"

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

def generate_4_query(centroids, golds, forward_1_edges=True, forward_2_edges=True, forward_3_edges=True, forward_4_edges=True):
    centroid_symbol = "s"
    gold_symbol = "o"

    first_edge_string = "?s ?r1 ?e" if forward_1_edges else "?e ?r1 ?s"
    second_edge_string = "?e ?r2 ?i" if forward_2_edges else "?i ?r2 ?e"
    third_edge_string = "?i ?r3 ?e2" if forward_3_edges else "?e2 ?r3 ?i"
    fourth_edge_string = "?e2 ?r4 ?o" if forward_4_edges else "?o ?r4 ?e2"

    query = "PREFIX ns: <http://rdf.freebase.com/ns/>"
    query += "\n\nselect * where {"
    query += "\n\t" + first_edge_string + " ."
    query += "\n\t" + second_edge_string + " ."
    query += "\n\t" + third_edge_string + " ."
    query += "\n\t" + fourth_edge_string + " ."
    query += "\n\tvalues ?" + centroid_symbol + " { " + " ".join(centroids) + " }"
    query += "\n\tvalues ?" + gold_symbol + " { " + " ".join([format_string_for_freebase(g) for g in golds]) + " }"

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
        yield r["r1"]["value"], r["r2"]["value"]

def get_3_paths(centroids, golds):
    for permutation in itertools.product([True, False], repeat=3):
        #print(permutation)
        yield from get_3_paths_internal(centroids, golds, permutation[0], permutation[1], permutation[2])

def get_3_paths_internal(centroids, golds, forward_1, forward_2, forward_3):
    query = generate_2_query_through_event(centroids, golds, forward_1, forward_2, forward_3)
    #print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        yield r["r1"]["value"], r["r2"]["value"], r["r3"]["value"]

def get_4_paths(centroids, golds):
    for permutation in itertools.product([True, False], repeat=4):
        yield from get_4_paths_internal(centroids, golds, permutation[0], permutation[1], permutation[2])

def get_4_paths_internal(centroids, golds, forward_1, forward_2, forward_3, forward_4):
    query = generate_4_query(centroids, golds, forward_1, forward_2, forward_3, forward_4)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        yield r["r1"]["value"], r["r2"]["value"], r["r3"]["value"], r["r4"]["value"]

path_counter = {}
final_edge_counter = {}
all_edge_counter = {}
edges_required_counter = {1:0, 2:0, 3:0}

path_file = open(args.file + ".path_count")
final_edge_file = open(args.file + ".final_edge_count")
all_edge_file = open(args.file + ".all_edge_count")
edges_required_file = open(args.file + ".edges_required_count")

counter = 0
for gold, sentence in zip(gold_reader.parse_file(args.file), sentence_reader.parse_file(args.file)):
    print(counter)
    counter += 1
    found = False

    if counter == 3:
        break

    for edge in get_1_paths(sentence, gold):
        edges_required_counter[1] += 1
        found = True

        if edge not in path_counter:
            path_counter[edge] = 0
        path_counter[edge] += 1

        if edge not in final_edge_counter:
            final_edge_counter[edge] = 0
        final_edge_counter[edge] += 1

        if edge not in all_edge_counter:
            all_edge_counter[edge] = 0
        all_edge_counter[edge] += 1

    for edge_1,edge_2 in get_2_paths(sentence, gold):
        edge = str(edge_1) + " " + str(edge_2)

        if not found:
            edges_required_counter[1] += 1
            found = True

        if edge not in path_counter:
            path_counter[edge] = 0
        path_counter[edge] += 1

        if edge_2 not in final_edge_counter:
            final_edge_counter[edge] = 0
        final_edge_counter[edge] += 1

        for edge in [edge_1, edge_2]:
            if edge not in all_edge_counter:
                all_edge_counter[edge] = 0
            all_edge_counter[edge] += 1

    for edge_1,edge_2, edge_3 in get_3_paths(sentence, gold):
        edge = str(edge_1) + " " + str(edge_2+ " " + str(edge_3))

        if not found:
            edges_required_counter[1] += 1
            found = True

        if edge not in path_counter:
            path_counter[edge] = 0
        path_counter[edge] += 1

        if edge_3 not in final_edge_counter:
            final_edge_counter[edge] = 0
        final_edge_counter[edge] += 1

        for edge in [edge_1, edge_2, edge_3]:
            if edge not in all_edge_counter:
                all_edge_counter[edge] = 0
            all_edge_counter[edge] += 1

for k, v in path_counter:
    print(str(k) + "\t" + str(v), file=path_file)

for k, v in final_edge_counter:
    print(str(k) + "\t" + str(v), file=final_edge_file)

for k, v in all_edge_counter:
    print(str(k) + "\t" + str(v), file=all_edge_file)

for k, v in edges_required_counter:
    print(str(k) + "\t" + str(v), file=edges_required_file)

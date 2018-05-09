import argparse
import json

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
parser.add_argument('--json_files', type=str, help='The location of the .json-files to be included, separated by |')
args = parser.parse_args()

dep_dict = {}

for dep_file in args.json_files.split("|"):
    with open(dep_file, "r") as data_file:
        for line in data_file:
            json_line = json.loads(line)
            key = json_line["original"].replace(" ", "")
            dep_dict[key] = ([(x["entities"], x["dependency_lambda"]) for x in json_line["forest"]])

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

            entities_and_graphs = dep_dict["".join(sentence).lower()]

            for entities, graph in entities_and_graphs:
                for entity in entities:
                    if "score" in entity:
                        entity_parts = [str(entity["index"]), str(entity["end"]), entity["entity"], str(entity["score"])]
                        print("\t".join(entity_parts))

                print("")

                for clause in graph[0]:
                    clause_parts = [None, None, None]
                    clause_parts[0] = clause.split("(")[0].strip()
                    subject = clause.split("(")[1].split(",")

                    if len(subject) == 1:
                        clause_parts[1] = subject[0].strip()
                        clause_parts[2] = subject[0].strip()[:-1]
                    else:
                        clause_parts[1] = subject[0].strip()
                        clause_parts[2] = subject[1].strip()[:-1]

                    print("\t".join(clause_parts))

                print("")

            sentence = []
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

import argparse
import json

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
parser.add_argument('--json_files', type=str, help='The location of the .json-files to be included, separated by |')
args = parser.parse_args()

dep_dict = {}

seen_relations = {}

for dep_file in args.json_files.split("|"):
    with open(dep_file, "r") as data_file:
        for line in data_file:
            json_line = json.loads(line)
            key = json_line["original"].replace(" ", "")
            relations = json_line["goldRelations"] if "goldRelations" in json_line else [None]
            dep_dict[key] = relations

            for relation_pair in relations:
                if relation_pair is None:
                    continue

                left_relation = relation_pair["relationLeft"]
                if left_relation not in seen_relations:
                    seen_relations[left_relation] = True

                if left_relation.endswith(".1") or left_relation.endswith(".2"):
                    continue
                else:
                    right_relation = relation_pair["relationRight"]
                    if right_relation not in seen_relations:
                        seen_relations[right_relation] = True

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

            for relation_pair in entities_and_graphs:
                if relation_pair is None:
                    continue

                left_relation = relation_pair["relationLeft"]
                right_relation = relation_pair["relationRight"]

                print(left_relation + "\t" + right_relation + "\t" + str(relation_pair["score"]))

            print("")

            sentence = []
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

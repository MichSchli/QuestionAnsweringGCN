import argparse
import json

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
parser.add_argument('--json_files', type=str, help='The location of the .json-files to be included, separated by |')
args = parser.parse_args()

dep_dict = {}

for dep_file in args.json_files.split("|"):
    with open(dep_file) as data_file:
        for line in data_file:
            json_line = json.loads(line)
            key = json_line["original"].replace(" ", "")
            print([x["dependency_lambda"] for x in json_line["forest"]])
            break

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
            print("".join(sentence))
            break
            sentence = []
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

    if len(sentence) > 0:
        print("".join(sentence))

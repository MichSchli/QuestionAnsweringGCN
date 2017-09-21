import json
import argparse
import string

parser = argparse.ArgumentParser(description='Parses webquestions files with entity annotations to our internal format.')
parser.add_argument('--input_file', type=str, help='The location of the .json-file to be parsed')
parser.add_argument('--entity_file', type=str, help='The location of the associated entities')
args = parser.parse_args()

sentences = json.load(open(args.input_file))
entities = open(args.entity_file)

entity_line = entities.readline().strip().split('\t')

for i,line in enumerate(sentences):
    sentence_matrix = []
    sentence_entity_lines = []

    utterance = line["utterance"].strip()

    # Split punctuation:
    for c in string.punctuation:
        if utterance.endswith(c):
            utterance = utterance[:-1] + " " + c

    words = utterance.split(" ")
    char_counter = 0
    for word_counter,word in enumerate(words):
        word_vector = ["_"] * 5
        word_vector[0] = str(word_counter)
        word_vector[1] = word

        word_counter += 1
        sentence_matrix.append(word_vector)

    entity_matrix = []
    while entity_line[0].endswith(str(i)):
        entity_words = line["utterance"][int(entity_line[2]):int(entity_line[2])+int(entity_line[3])]
        words_before = line["utterance"][:int(entity_line[2])].count(" ")
        words_in = entity_words.count(" ")

        entity_vector = ["_"]*4
        entity_vector[0] = str(words_before)
        entity_vector[1] = str(words_before + words_in)
        entity_vector[2] = entity_line[4][1:].replace("/", ".")
        entity_vector[3] = entity_line[6]

        entity_matrix.append(entity_vector)

        entity_line = entities.readline().strip().split('\t')

    answer_matrix = []

    answer_descriptions = line["targetValue"][19:-2]
    answer_descriptions = answer_descriptions.replace("\"", "")
    answer_descriptions = answer_descriptions.split(") (description ")
    for description in answer_descriptions:
        answer_matrix.append(["_", description])

    if i > 0:
        print("")

    print("\n".join(["\t".join(line) for line in sentence_matrix]))
    print("")
    if len(entity_matrix) > 0:
        print("\n".join(["\t".join(line) for line in entity_matrix]))
    print("")
    print("\n".join(["\t".join(line) for line in answer_matrix]))
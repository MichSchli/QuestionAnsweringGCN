import json
import argparse
import string

parser = argparse.ArgumentParser(description='Parses webquestions files with entity annotations to our internal format.')
parser.add_argument('--input_file', type=str, help='The location of the .txt-file to be parsed')
parser.add_argument('--output_file', type=str, help='The location of the .conll-file to be produced')
parser.add_argument('--entity_file', type=str, help='The location of the associated entities')
parser.add_argument('--link_file', type=str, help='The location of the associated link file')
args = parser.parse_args()

i = 0

for sentence, entities, link in zip(open(args.input_file), open(args.entity_file), open(args.link_file)):
    sentence_matrix = []
    entity_matrix = []
    answer_matrix = []

    utterance = sentence.strip().split("\t")[3]

    words = utterance.split(" ")
    char_counter = 0
    for word_counter, word in enumerate(words):
        word_vector = ["_"] * 5
        word_vector[0] = str(word_counter)
        word_vector[1] = word

        word_counter += 1
        sentence_matrix.append(word_vector)

    link_utterance = link.strip().split("\t")[-1]
    before = link_utterance.find("#head_entity#") - 1
    after = len(link_utterance) - before - len("#head_entity# ")
    words_before = len(utterance[:before].split(" "))
    words_in = len(utterance[before+1:-after].split(" "))

    counter = 0
    for entity in entities.strip().split('\t')[1:]:
        counter += 1
        if counter >= 10:
            break

        entity_vector = ["_"] * 4
        entity_vector[0] = str(words_before)
        entity_vector[1] = str(words_before + words_in)
        entity_vector[2] = entity.split(" ")[0]
        entity_vector[3] = entity.split(" ")[1]

        entity_matrix.append(entity_vector)

    answer_matrix = []
    answer_matrix.append([sentence.strip().split("\t")[2], sentence.strip().split("\t")[2]])

    if i > 0:
        print("")
    i += 1

    print("\n".join(["\t".join(line) for line in sentence_matrix]))
    print("")
    if len(entity_matrix) > 0:
        print("\n".join(["\t".join(line) for line in entity_matrix]))
    print("")
    if len(answer_matrix) > 0:
        print("\n".join(["\t".join(line) for line in answer_matrix]))

import json
import argparse

parser = argparse.ArgumentParser(description='Parses json-formatted SPADES files to our internal format.')
parser.add_argument('--input_file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

for i,line in enumerate(open(args.input_file)):
    line = json.loads(line)
    entity_indexes = [e['index'] for e in line['entities']]
    entity_index_dict = {e['index']:(e['entity'], e['score']) for e in line['entities']}

    sentence_matrix = []
    entity_matrix = []
    word_counter = 0

    for j,word in enumerate(line['words']):
        if j in entity_indexes:
            components = word["word"].split("_")

            entity_vector = ["_"]*4
            entity_vector[0] = str(j)
            entity_vector[1] = str(j + len(components) - 1)
            entity_vector[2] = entity_index_dict[j][0]
            entity_vector[3] = str(entity_index_dict[j][1])

            entity_matrix.append(entity_vector)

            first = True
            for component in components:
                word_vector = ["_"] * 5
                word_vector[0] = str(word_counter)
                word_vector[1] = component
                word_vector[2] = component
                word_vector[3] = word["pos"]
                word_vector[4] = word["ner"]

                sentence_matrix.append(word_vector)

                first = False
                word_counter += 1
        else:
            word_vector = ["_"] * 5
            word_vector[0] = str(word_counter)
            word_vector[1] = word["word"]
            word_vector[2] = word["lemma"]
            word_vector[3] = word["pos"]
            word_vector[4] = word["ner"]
            word_counter += 1
            sentence_matrix.append(word_vector)

    answer_matrix = []
    for mid, description in zip(line['answerSubset'], line['answerString']):
        answer_matrix.append([mid, description])

    if i > 0:
        print("")

    print("\n".join(["\t".join(line) for line in sentence_matrix]))
    print("")
    print("\n".join(["\t".join(line) for line in entity_matrix]))
    print("")
    print("\n".join(["\t".join(line) for line in answer_matrix]))
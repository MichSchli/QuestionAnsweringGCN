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
    word_counter = 0

    for j,word in enumerate(line['words']):
        if j in entity_indexes:
            components = word["word"].split("_")
            first = True
            for component in components:
                word_vector = ["_"] * 8
                word_vector[0] = str(word_counter)
                word_vector[1] = component
                word_vector[2] = component
                word_vector[3] = word["pos"]
                word_vector[4] = word["ner"]
                word_vector[5] = "B" if first else "I"
                word_vector[6] = entity_index_dict[j][0]
                word_vector[7] = str(entity_index_dict[j][1])

                sentence_matrix.append(word_vector)

                first = False
                word_counter += 1
        else:
            word_vector = ["_"] * 8
            word_vector[0] = str(word_counter)
            word_vector[1] = word["word"]
            word_vector[2] = word["lemma"]
            word_vector[3] = word["pos"]
            word_vector[4] = word["ner"]
            word_vector[5] = "O"
            word_counter += 1
            sentence_matrix.append(word_vector)

    answer_matrix = []
    for mid, description in zip(line['answerSubset'], line['answerString']):
        answer_matrix.append([mid, description])

    if i > 0:
        print("")

    print("\n".join(["\t".join(line) for line in sentence_matrix]))
    print("")
    print("\n".join(["\t".join(line) for line in answer_matrix]))
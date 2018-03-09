import argparse
import spacy

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

parser = spacy.load('en')

with open(args.input_file) as data_file:
    reading_sentence = True
    reading_entities = False
    sentence = []
    for line in data_file:
        line = line.strip()

        if line and reading_sentence:
            parts = line.split("\t")
            sentence.append(parts + ["_"])
        elif not line and reading_sentence:
            doc = parser(" ".join([w[1] for w in sentence]))
            out_sentence = []
            pointer = 0
            current_word = ""
            splits = {}
            for i,token in enumerate(doc):
                word = [None]*6
                word[0] = str(i)
                word[1] = token.text
                word[2] = token.lemma_
                word[3] = token.tag_
                word[4] = token.dep_
                word[5] = str(token.head.i if token.head.i != i else 0)
                out_sentence.append(word)

                current_word += token.text
                if current_word != sentence[pointer][1]:
                    if pointer not in splits:
                        splits[pointer] =[]
                    splits[pointer].append(i)
                else:
                    if pointer in splits:
                        splits[pointer].append(i)
                    pointer += 1
                    current_word = ""

            print("\n".join(["\t".join(w) for w in out_sentence]))
            sentence = []
            reading_sentence = False
            reading_entities = True
            print("")
        elif not line and reading_entities:
            reading_entities = False
            print("")
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True
            print("")
        elif reading_entities:
            parts = line.split("\t")
            for k,v in splits.items():
                print(parts)
                if k <= int(parts[0]):
                    parts[0] = str(int(parts[0]) + len(v) - 1)
                if k <= int(parts[1]):
                    parts[1] = str(int(parts[1]) + len(v) - 1)
            print("\t".join(parts))
        else:
            print(line)

    if len(sentence) > 0:
        doc = parser(" ".join([w[1] for w in sentence]))
        for i, token in enumerate(doc):
            sentence[i][2] = token.lemma_
            sentence[i][3] = token.tag_
            sentence[i][4] = token.dep_
            sentence[i][5] = str(token.head.i if token.head.i != i else 0)
        print("\n".join(["\t".join(w) for w in sentence]))
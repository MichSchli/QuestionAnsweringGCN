import argparse

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--input_file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

with open(args.input_file) as data_file:
    reading_sentence = True
    reading_entities = False
    sentence = []
    for line in data_file:
        line = line.strip()

        if line and reading_sentence:
            parts = line.split("\t")
            sentence.append(parts[1])
        elif not line and reading_sentence:
            print(" ".join(sentence))
            sentence = []
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

    if len(sentence) > 0:
        print(" ".join(sentence))
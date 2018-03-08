import json
import argparse
import string

parser = argparse.ArgumentParser(description='Flattens a conll file to individual sentences.')
parser.add_argument('--structure', type=str, help='The location of the .conll-file with sentence structure etc')
parser.add_argument('--flat', type=str, help='The location of the flat file to structure')
args = parser.parse_args()

flat_file = open(args.flat)
flat_sentence = flat_file.readline().strip().split(" ")
word_pointer = 0

with open(args.structure) as data_file:
    reading_sentence = True
    reading_entities = False
    for line in data_file:
        line = line.strip()

        if line and reading_sentence:
            parts = line.split("\t")
            parts[1] = flat_sentence[word_pointer]
            word_pointer += 1
            print("\t".join(parts))
            continue
        elif not line and reading_sentence:
            flat_sentence = flat_file.readline().strip().split(" ")
            word_pointer = 0
            sentence = []
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True

        print(line)

    if len(sentence) > 0:
        print(" ".join(sentence))
import argparse
import json
import sys


def parse_json_line(json_line):
    return ' '.join([parse_json_word(word) for word in json_line['words']])


def parse_json_word(word):
    return word['word'] + '|' + word['pos'] + '|' + word['ner']


def parse_from_file(filename):
    with open(filename) as data_file:
        for line in (data_file):
            json_line = json.loads(line)
            print(parse_json_line(json_line))


def parse_from_console():
    for line in sys.stdin:
        json_line = json.loads(line)
        print(parse_json_line(json_line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Formats and yields json data to stdout for easyccg parsing.')
    parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')

    args = parser.parse_args()

    if args.file is not None:
        parse_from_file(args.file)
    else:
        parse_from_console()


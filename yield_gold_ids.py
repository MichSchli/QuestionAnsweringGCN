import argparse
from preprocessing.read_json_files import JsonReader

parser = argparse.ArgumentParser(description='Yields gold entities to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')
args = parser.parse_args()

json_reader = JsonReader(output="gold")
for line in json_reader.parse_file(args.file):
    print(line)
from grounding.json_to_candidate_neighborhood import parse_from_file
import argparse

#from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface

#iface = FreebaseInterface()
#results = iface.get_neighborhood(["m.014zcr", "m.0q0b4"], edge_limit=4000, hops=2)
#print(results[0].shape)
#print(results[1].shape)

parser = argparse.ArgumentParser(description='Yields candidate graphs to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')

args = parser.parse_args()

if args.file is not None:
    parse_from_file(args.file)
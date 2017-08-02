from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface
from grounding.json_to_candidate_neighborhood import CandidateNeighborhoodGenerator
import argparse

#from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface

#iface = FreebaseInterface()
#results = iface.get_neighborhood(["m.014zcr", "m.0q0b4"], edge_limit=4000, hops=2)
#print(results[0].shape)
#print(results[1].shape)
from preprocessing.read_spades_files import JsonReader

parser = argparse.ArgumentParser(description='Yields candidate graphs to stdout.')
parser.add_argument('--file', type=str, help='The location of the .json-file to be parsed')

args = parser.parse_args()

fb = FreebaseInterface()
jp = JsonReader()
cng = CandidateNeighborhoodGenerator(fb, jp)

if args.file is not None:
    for g in cng.parse_file(args.file):
        print(g.vertices)
        print(g.edges[0])
        print("================")

import argparse
import json
import sys
from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface
from model.candidate_graph import CandidateGraph
from preprocessing.read_spades_files import JsonReader
import numpy as np


class CandidateNeighborhoodGenerator:

    freebase_interface = None
    neighborhood_search_scope = None
    max_candidates = None
    json_reader = None

    def __init__(self, freebase_interface, json_reader, neighborhood_search_scope=2, max_candidates=10000):
        self.freebase_interface = freebase_interface
        self.neighborhood_search_scope = neighborhood_search_scope
        self.max_candidates = max_candidates
        self.json_reader = json_reader

    def parse_file(self, filename):
        for sentence_entities in self.json_reader.parse_file(filename, output="entities", print_progress=True):
            vertices, edges = self.freebase_interface.get_neighborhood(sentence_entities, edge_limit=self.max_candidates, hops=self.neighborhood_search_scope)
            candidate_graph = CandidateGraph(sentence_entities, vertices, edges)
            yield candidate_graph

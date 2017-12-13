from collections import defaultdict
from time import sleep

from database_interface.data_interface.FreebaseInterface import FreebaseInterface
from database_interface.expansion_strategies.only_freebase_element_strategy import OnlyFreebaseExpansionStrategy
from database_interface.hypergraph_interface import HypergraphInterface
from database_interface.indexed_interface import IndexedInterface
from experiment_construction.candidate_generator_construction.candidate_generator_cache import CandidateGeneratorCache
from experiment_construction.candidate_generator_construction.neighborhood_candidate_generator import \
    NeighborhoodCandidateGenerator
from experiment_construction.index_construction.indexes.freebase_indexer import FreebaseIndexer
from experiment_construction.index_construction.indexes.freebase_relation_indexer import FreebaseRelationIndexer
from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer
from helpers.read_conll_files import ConllReader
import numpy as np

database_interface = FreebaseInterface()
expansion_strategy = OnlyFreebaseExpansionStrategy()

prefix = "http://rdf.freebase.com/ns/"
disk_cache = "/datastore/michael_cache/webquestions.1neighbors.cache"

database = HypergraphInterface(database_interface, expansion_strategy, prefix=prefix)

candidate_generator = NeighborhoodCandidateGenerator(database, neighborhood_search_scope=1,
                                                     extra_literals=True)

e_indexer = LazyIndexer((40000,1))
r_indexer = FreebaseRelationIndexer((6000,1), 10)

database = IndexedInterface(database, e_indexer, r_indexer)
candidate_generator = CandidateGeneratorCache(candidate_generator,
                                              disk_cache=disk_cache)

train_file_iterator = ConllReader("data/webquestions/train.split.conll", entity_prefix=prefix)
epoch_iterator = train_file_iterator.iterate()
epoch_iterator = candidate_generator.enrich(epoch_iterator)


def project_from_name_wrapper(iterator, skip=True):
    for example in iterator:
        names = example["gold_entities"]
        graph = example["neighborhood"]
        name_projection_dictionary = graph.get_inverse_name_connections(names)

        gold_list = []
        for name, l in name_projection_dictionary.items():
            if len(l) > 0:
                gold_list.extend(l)
            elif graph.has_index(name):
                gold_list.append(graph.to_index(name))

        # TODO CHECK SOMEWHERE ELSE
        if len(gold_list) == 0:
            # print("name " + str(names) + " does not match anything, discarding")
            if not skip:
                yield example

            continue

        gold_list = np.array(gold_list).astype(np.int32)
        # print(example["neighborhood"].entity_vertices.shape[0])
        # print("projected " + str(example["gold_entities"]) + " to " + str(gold_list))
        example["gold_entities"] = gold_list
        yield example

epoch_iterator = list(project_from_name_wrapper(epoch_iterator))

counter = defaultdict(int)

for example in epoch_iterator:
    for g in example["gold_entities"]:
        path = example["neighborhood"].get_paths_to_neighboring_centroid(g)
        if len(path) == 4:
            label = path[1] + path[2]
        else:
            label = path[1] + path[2] + path[4] + path[5]

        print(label)

    sleep(3)

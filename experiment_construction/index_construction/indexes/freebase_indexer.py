import struct

import numpy as np

from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer


class FreebaseIndexer:

    vectors = None
    indexer = None

    def __init__(self):
        self.dimension = 1000
        self.load_file()

    def get_dimension(self):
        return self.dimension

    def index(self, elements):
        local_map = np.empty(elements.shape, dtype=np.int32)

        for i, element in enumerate(elements):
            local_map[i] = self.index_single_element(element)

        return local_map

    def index_single_element(self, element):
        if element not in self.indexer.global_map:
            return 0
        else:
            return self.indexer.global_map[element]

    def retrieve_vector(self, index):
        return self.get_all_vectors()[index]

    def get_all_vectors(self):
        return self.vectors

    def load_file(self):
        file_string = "/home/mschlic1/GCNQA/data/embeddings/freebase-vectors-skipgram1000.bin"

        f = open(file_string, "rb")
        vocab_size = ""
        while True:
            c = f.read(1)
            if ord(c) == ord(" "):
                break
            else:
                vocab_size += chr(ord(c))

        vocab_size = int(vocab_size)
        print(vocab_size)
        self.vectors = np.zeros((vocab_size+1, self.dimension), dtype=np.float32)
        vocab = [None]*(vocab_size+1)

        f.read(5)

        word = True
        counter = 1

        current_mid = ""
        while True:
            if word:
                c = f.read(1)
                if not c:
                    break
                if ord(c) == ord(" "):
                    vocab[counter] = current_mid
                    current_mid = ""
                    word = False

                    if counter % 10000 == 0:
                        print(counter)
                else:
                    current_mid += chr(ord(c))
            else:
                n = f.read(4000)
                number = np.array(struct.unpack("1000f", n))

                self.vectors[counter] = number

                counter += 1
                word = True

        self.indexer = LazyIndexer((vocab_size, self.dimension))

        for word in vocab[1:]:
            self.indexer.index_single_element("http://rdf.freebase.com/ns/" + word[1:].replace("/", "."))

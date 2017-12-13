import struct

import numpy as np

from experiment_construction.index_construction.abstract_indexer import AbstractIndexer
from experiment_construction.index_construction.indexes.lazy_indexer import LazyIndexer


class FreebaseIndexer(AbstractIndexer):

    vectors = None
    indexer = None

    def __init__(self):
        self.dimension = 1000
        vocab = self.load_file()
        inner = LazyIndexer((len(vocab), self.dimension))
        AbstractIndexer.__init__(self, inner)

        for word in vocab:
            self.inner.index_single_element("http://rdf.freebase.com/ns/" + word[1:].replace("/", "."))

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
        self.vectors = np.zeros((vocab_size+1, self.dimension), dtype=np.float32)
        vocab = [None]*(vocab_size)

        f.read(5)

        word = True
        counter = 0

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
                else:
                    current_mid += chr(ord(c))
            else:
                n = f.read(4000)
                number = np.array(struct.unpack("1000f", n))

                self.vectors[counter+1] = number

                counter += 1
                word = True

        return vocab

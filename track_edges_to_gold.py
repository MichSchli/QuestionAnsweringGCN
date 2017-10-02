#sparql.setQuery("""
#PREFIX ns: <http://rdf.freebase.com/ns/>

#select count(*) where {
#?s ?r ?o .
#values ?s { ns:m.03_r3 }
#}
#"""
import argparse

from preprocessing.read_conll_files import ConllReader

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

gold_reader = ConllReader(output="gold")
sentence_reader = ConllReader()

for gold, sentence in zip(gold_reader.parse_file(args.file), sentence_reader.parse_file(args.file)):
    print(gold)
    print(sentence)
    exit()
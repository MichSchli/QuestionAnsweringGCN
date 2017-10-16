import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description='Yields pairs of prediction from a strategy and gold to stdout.')
parser.add_argument('--in_file', type=str, help='The location of the .conll-file to be parsed')
parser.add_argument('--out_file_1', type=str, help='The location of the .conll-file to be parsed')
parser.add_argument('--out_file_2', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

split_size = .8
parts = 3

elements = []

counter = 0
holder = [[]]
for line in open(args.in_file):
    line = line.strip()
    if line:
        holder[-1].append(line)
    else:
        counter += 1
        if counter == parts:
            elements.append(["\n".join(part) for part in holder])
            holder = [[]]
            counter = 0
        else:
            holder.append([])

partition = int(split_size*len(elements))
np.random.shuffle(elements)

split_1 = elements[:partition]
split_2 = elements[partition:]

out_file_1 = open(args.out_file_1, "w+")
print("\n\n".join(["\n\n".join(e) for e in split_1]), file=out_file_1)

out_file_2 = open(args.out_file_2, "w+")
print("\n\n".join(["\n\n".join(e) for e in split_2]), file=out_file_2)

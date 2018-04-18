import argparse

parser = argparse.ArgumentParser(description='Adds alignment groups to an input file on conll format.')
parser.add_argument('--file', type=str, help='The location of the .conll-file to be parsed')
args = parser.parse_args()

read_mode = "sentence"
examples = [[[],[],[]]]
for line in open(args.file, 'r'):
    line = line.strip()

    if read_mode == "sentence":
        if not line:
            read_mode = "entity"
            alignment_groups = {}
            alignment_group_counter = 0
        else:
            examples[-1][0].append(line)
        continue

    if read_mode == "entity":
        if not line:
            read_mode = "gold"
        else:
            parts = line.split('\t')
            alignment_group_id = str(parts[0:2])

            if alignment_group_id not in alignment_groups:
                alignment_groups[alignment_group_id] = alignment_group_counter
                alignment_group_counter += 1

            parts.append(str(alignment_groups[alignment_group_id]))
            examples[-1][1].append('\t'.join(parts))
        continue

    if read_mode == "gold":
        if not line:
            read_mode = "sentence"
            examples.append([[],[],[]])
        else:
            examples[-1][2].append(line)

        continue

if examples[-1] == [[],[],[]]:
    examples = examples[:-1]

def print_to_file(data, file):
    first = True
    for s_matrix, e_matrix, t_matrix in data:
        if first:
            first = False
        else:
            print("", file=file)

        print("\n".join([line for line in s_matrix]), file=file)
        print("", file=file)
        if len(e_matrix) > 0:
            print("\n".join([line for line in e_matrix]), file=file)
        print("", file=file)
        print("\n".join([line for line in t_matrix]), file=file)

outfile = open(args.file, "w")
print_to_file(examples, outfile)
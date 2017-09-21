import argparse
import random

parser = argparse.ArgumentParser(description='Creates a toy dataset.')
parser.add_argument('--graph', type=str, help='The location of the .graph-file to contain the graph')
parser.add_argument('--train_file', type=str, help='The location of the .conll file to contain train data')
parser.add_argument('--test_file', type=str, help='The location of the .conll file to contain test data')
args = parser.parse_args()

edges = []

stop = False

entity_vertices = []
event_vertices = []
literal_vertices = []

sentences = []
sentence_entities = []

while not stop:
    print("Input pattern:")
    pattern = input()
    if pattern.strip() == "stop":
        stop = True
        continue

    print("Input sentence:")
    from_console = input()

    if from_console.strip() == "stop":
        stop = True
        continue

    words = from_console.strip().split(' ')

    sentences.append(words)

    if pattern == "1":
        s = words[0]
        v = words[1]
        o = words[2]

        s_type = "entity" if s.endswith("_") else "literal"
        o_type = "entity" if o.endswith("_") else "literal"

        sentence_entities.append([[0, s[:-1]], [2, o[:-1]]])

        edge = [s[:-1], v, o[:-1], s_type, o_type]
        edges.append(edge)
    elif pattern == "2":
        s = words[0]
        v = words[1]
        o = words[2]
        p = words[3]
        m = words[4]

        s_type = "entity" if s.endswith("_") else "literal"
        o_type = "entity" if o.endswith("_") else "literal"
        m_type = "entity" if m.endswith("_") else "literal"

        sentence_entities.append([[0, s[:-1]], [2, o[:-1]], [4, m[:-1]]])

        event = "e_" + str(len(event_vertices))
        event_vertices.append(event)

        edges.append([s[:-1], v+".subject", event, s_type, "event"])
        edges.append([o[:-1], v+".object", event, o_type, "event"])
        edges.append([m[:-1], p, event, m_type, "event"])
    elif pattern == "3":
        s = words[0]
        v = words[1]
        o = words[2]
        p1 = words[3]
        m1 = words[4]
        p2 = words[5]
        m2 = words[6]

        s_type = "entity" if s.endswith("_") else "literal"
        o_type = "entity" if o.endswith("_") else "literal"
        m1_type = "entity" if m1.endswith("_") else "literal"
        m2_type = "entity" if m2.endswith("_") else "literal"

        sentence_entities.append([[0, s[:-1]], [2, o[:-1]], [4, m[:-1]], [6, m[:-1]]])

        event = "e_" + str(len(event_vertices))
        event_vertices.append(event)

        edges.append([s[:-1], v+".subject", event, s_type, "event"])
        edges.append([o[:-1], v+".object", event, o_type, "event"])
        edges.append([m1[:-1], p1, event, m1_type, "event"])
        edges.append([m2[:-1], p2, event, m2_type, "event"])
    elif pattern == "4":
        s = words[0]
        v = ".".join(words[2:-1])
        o = words[-1]

        s_type = "entity" if s.endswith("_") else "literal"
        o_type = "entity" if o.endswith("_") else "literal"

        sentence_entities.append([[0, s[:-1]], [len(words)-1, o[:-1]]])

        edge = [s[:-1], v, o[:-1], s_type, o_type]
        edges.append(edge)

    continue

    if pattern == "2" or pattern == "3":
        event = "e_" + str(len(event_vertices))
        event_vertices.append(event)

    v1 = None
    v2 = None
    v3 = None
    v4 = None

    for word in words:
        # Entity:
        if word.endswith("_"):
            entity = word[:-1]
            if entity not in entity_vertices:
                entity_vertices.append(entity)
        # Literal:
        elif word.endswith("!"):
            literal = word[:-1]
            if literal not in entity_vertices:
                literal_vertices.append(literal)

dataset = []

for sentence, entities in zip(sentences, sentence_entities):
    entity_indexes = [e[0] for e in entities]
    for idx in entity_indexes:
        sentence[idx] = sentence[idx][:-1]

    for target in entities:
        sentence_matrix = []

        for i,word in enumerate(sentence):
            word_vector = ["_"]*5
            word_vector[0] = str(i)
            word_vector[1] = word if i != target[0] else "_blank_"
            sentence_matrix.append(word_vector)

        entity_matrix = []

        for e in entities:
            if e[0] != target[0]:
                entity_vector = [None]*4
                entity_vector[0] = str(e[0])
                entity_vector[1] = str(e[0])
                entity_vector[2] = e[1]
                entity_vector[3] = "1.0"

                entity_matrix.append(entity_vector)

        target_matrix = [[target[1], target[1]]]

        dataset.append((sentence_matrix, entity_matrix, target_matrix))


random.shuffle(dataset)
split = int(len(dataset)*0.8)

training_data = dataset[:split]
test_data = dataset[split:]

graph_file = open(args.graph, "w+")
train_file = open(args.train_file, "w+")
test_file = open(args.test_file, "w+")

for edge in edges:
    print(",".join(edge), file=graph_file)


def print_to_file(data, file):
    for s_matrix, e_matrix, t_matrix in data:
        print("\n".join(["\t".join(line) for line in s_matrix]), file=file)
        print("", file=file)
        print("\n".join(["\t".join(line) for line in e_matrix]), file=file)
        print("", file=file)
        print("\n".join(["\t".join(line) for line in t_matrix]), file=file)


print_to_file(training_data, train_file)
print_to_file(test_data, test_file)

print("I made a dataset!")

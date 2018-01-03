import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description='Creates a toy dataset.')
parser.add_argument('--graph', type=str, help='The location of the .graph-file to contain the graph.')
parser.add_argument('--train_file', type=str, help='The location of the .conll file to contain train data.')
parser.add_argument('--test_file', type=str, help='The location of the .conll file to contain test data.')
parser.add_argument('--append', action="store_true", help='Decides whether to append instead of overwriting.')
args = parser.parse_args()


'''
Defines a radix tree structure to match input sentence and select patterns.
'''
class PatternTrie:

    word = None
    next = None
    identifier = None

    def __init__(self, word):
        self.word = word
        self.next = {}

    '''
    Insert a list of words in the trie recursively:
    '''
    def insert(self, word_list, identifier):
        if len(word_list) == 1:
            self.identifier = identifier
            return

        self.insert_in_subtree(identifier, word_list[1:])

    '''
    Internal method to insert a list of words under the correct subtree:
    '''
    def insert_in_subtree(self, identifier, word_list):
        first_word = word_list[0]
        if first_word not in self.next:
            self.next[first_word] = PatternTrie(first_word)
        self.next[first_word].insert(word_list, identifier)

    '''
    Get the identifier associated with a pattern by recursively walking the trie, skipping control characters:
    '''
    def get_identifier(self, pattern):
        if len(pattern) == 1:
            return self.identifier

        next_word = pattern[1]

        result = None
        if next_word in self.next:
            result = self.next[next_word].get_identifier(pattern[1:])

        if result is None and "E" in self.next:
            result = self.next["E"].get_identifier(pattern[1:])

        if result is None and "R" in self.next:
            result = self.next["R"].get_identifier(pattern[1:])

        if result is None and "P" in self.next:
            result = self.next["P"].get_identifier(pattern[1:])

        if result is None and "L" in self.next:
            result = self.next["L"].get_identifier(pattern[1:])

        return result

    '''
    Debug print:
    '''
    def pretty_print(self, tabs=0):
        print("\t"*tabs + self.word + " / ID: " + str(self.identifier))
        for subtree in self.next.values():
            subtree.pretty_print(tabs=tabs+1)


'''
Defines a class constructing toy datasets based on patterns:
'''
class ToyDatasetMaker:

    patterns = [
        ("who does E R ? E",
         [["E1", "R1+s", "E2", "entity", "entity"]],
         ["who R1 -s E2 ? E1",
          "who is R1 -d by E1 ? E2"]
         ),
        ("who is the R of E ? E",
         [["E2", "R1+.of", "E1", "entity", "entity"]],
         ["who is R1 to E1 ? E2",
          "who is E2 the R1 of ? E1",
          "who is E2 R1 to ? E1"]
         ),
        ("who does E R for ? E",
         [["E1", "R1+s.for", "E2", "entity", "entity"]],
         ["who R1 -s for E2 ? E1",
          "who is E1 R1 -ing for ? E2",
          "who is R1 -ing for E2 ? E1"]
         ),
        ("who does E R with ? E",
         [["E1", "R1+s.with", "E2", "entity", "entity"],
          ["E2", "R1+s.with", "E1", "entity", "entity"]],
         ["who does E2 R1 with ? E1",
          "who R1 -s with E2 ? E1",
          "who R1 -s with E1 ? E2",
          "who is R1 -ing with E2 ? E1",
          "who is R1 -ing with E1 ? E2"]
         ),
        ("what does E R P ? E",
         [["E1", "R1+s.+P1", "E2", "entity", "entity"]],
         ["who R1 -s P1 E2 ? E1",
          "what is E1 R1 -ing P1 ? E2",
          "who is R1 -ing P1 E2 ? E1"]
         ),
        ("where was E R ? E",
         [["E1", "R1+.in", "E2", "entity", "entity"]],
         ["who was R1 in E2 ? E1"]
         ),
        ("when was E R ? L",
         [["E1", "R1+.in", "L1", "entity", "literal"]],
         ["who was R1 in L1 ? E1"]
         ),
        ("where does E R ? E",
         [["E1", "R1+.in", "E2", "entity", "entity"]],
         ["who R1 -s in E2 ? E1"]
         ),
        ("who did E R in L ? E",
         [["E1", "R1+.1", "e+#1", "entity", "event"],
          ["E2", "R1+.2", "e+#1", "entity", "event"],
          ["L1", "R1+.in", "e+#1", "literal", "event"]],
         ["when did E1 R1 E2 ? L1",
          "who did R1 E2 in L1 ? E1"]
         ),
        ("what did E R in L ? E",
         [["E1", "R1+ed.1", "e+#1", "entity", "event"],
          ["E2", "R1+ed.2", "e+#1", "entity", "event"],
          ["L1", "R1+ed.in", "e+#1", "literal", "event"]],
         ["when did E1 R1 E2 ? L1",
          "who R1 -ed E2 in L1 ? E1",
          "who R1 -ed E2 ? E1",
          "when was E2 R1 -ed ? L1"]
         ),
        ("who did E R E to in L ? E",
         [["E1", "R1+.1", "e+#1", "entity", "event"],
          ["E2", "R1+.2", "e+#1", "entity", "event"],
          ["E3", "R1+.3", "e+#1", "entity", "event"],
          ["L1", "R1+.in", "e+#1", "literal", "event"]],
         ["when did E1 R1 E2 to E3 ? L1",
          "who R1 -ed E2 to E3 ? E1",
          "who did E1 R1 E2 to ? E3",
          "when was E2 R1 -ed to E3 ? L1"]
         )
    ]

    pattern_trie = None
    global_counter = None

    global_edges = None
    global_questions = None

    def __init__(self):
        self.pattern_trie = PatternTrie("BEGIN")
        for i,pattern in enumerate(self.patterns):
            self.pattern_trie.insert(["BEGIN"] + pattern[0].split(" "), i)

        self.global_counter = {}

        self.global_edges = []
        self.global_questions = {}

    def process(self, input_sentence):
        input_sentence = input_sentence.strip().split(" ")
        pattern_id = self.identify_pattern(input_sentence)

        if pattern_id is None:
            print("Sentence could not be parsed.")
            return

        pattern = self.patterns[pattern_id]

        element_dictionary = self.extract_element_dictionary(input_sentence, pattern)
        edges = self.get_edges(element_dictionary, pattern)
        entities, targets = self.extract_entities_and_targets(input_sentence, pattern[0])

        alt_questions = self.get_alternative_questions(element_dictionary, pattern)

        alt_entities = []
        alt_targets = []

        for a_q, a_p in zip(alt_questions, pattern[2]):
            a_e, a_t = self.extract_entities_and_targets(a_q, a_p)
            alt_entities.append(a_e)
            alt_targets.append(a_t)

        all_questions = [self.postprocess_question(q) for q in [input_sentence] + alt_questions]
        all_entities = [entities] + alt_entities
        all_targets = [targets] + alt_targets

        self.global_edges.extend(edges)
        for question, entities, targets in zip(all_questions, all_entities, all_targets):
            question_string = " ".join(question)

            if question_string in self.global_questions:
                self.global_questions[question_string][2].update(targets)
            else:
                self.global_questions[question_string] = (question,entities,set(targets))

    def postprocess_question(self, question):
        o = []
        for w in question:
            if w == "?":
                o.append(w)
                break
            elif w[0] == "-":
                o[-1] += w[1:]
            else:
                o.append(w)

        return o

    def get_alternative_questions(self, element_dictionary, pattern):
        alts = []
        for alternative_sentence in pattern[2]:
            parts = alternative_sentence.split(" ")
            alts.append([])
            for part in parts:
                if part in element_dictionary:
                    alts[-1].append(element_dictionary[part])
                else:
                    alts[-1].append(part)

        return alts


    def get_edges(self, elements, pattern):
        edges = []
        nums = {}
        for prototype_edge in pattern[1]:
            edge = [None]*5

            for i in range(3):
                num = False

                prototype_parts = prototype_edge[i].split("+")
                edge_string = ""
                for part in prototype_parts:
                    if part in elements:
                        edge_string += elements[part]
                    elif part[0] == "#":
                        num = True
                        num_id = part[1:]
                    else:
                        edge_string += part

                if num:
                    if edge_string+num_id not in nums:
                        if edge_string not in self.global_counter:
                            self.global_counter[edge_string] = 0

                        self.global_counter[edge_string] += 1
                        nums[edge_string+num_id] = self.global_counter[edge_string]

                    edge_string += "_"+str(nums[edge_string+num_id])

                edge[i] = edge_string

            edge[3] = prototype_edge[3]
            edge[4] = prototype_edge[4]
            edges.append(edge)

        return edges

    def extract_element_dictionary(self, sentence, pattern):
        elements = {}
        element_type_counter = {}
        pattern_parts = pattern[0].split(" ")
        for i in range(len(sentence)):
            pattern_part = pattern_parts[i]
            if pattern_part in ["E", "R", "P", "L"]:
                if pattern_part not in element_type_counter:
                    element_type_counter[pattern_part] = 0
                element_type_counter[pattern_part] += 1

                elements[pattern_part + str(element_type_counter[pattern_part])] = sentence[i]

        return elements

    def extract_entities_and_targets(self, sentence, pattern):
        collect_target = False
        entities = []
        pattern_parts = pattern.split(" ")
        targets = []
        for i in range(len(sentence)):
            pattern_part = pattern_parts[i]
            if pattern_part[0] in ["E", "R", "P", "L"]:
                if pattern_part[0] == "E" and not collect_target:
                    entities.append([i, sentence[i]])
                elif collect_target:
                    targets.append(sentence[i])

            elif pattern_part == "?":
                collect_target = True
        return entities, targets

    def identify_pattern(self, sentence):
        pattern_id = self.pattern_trie.get_identifier(["BEGIN"]+sentence)
        return pattern_id

    def get_dataset_representation(self):
        dataset = []
        for question,entities,targets in self.global_questions.values():
            sentence_matrix = []

            for i, word in enumerate(question):
                word_vector = ["_"] * 5
                word_vector[0] = str(i)
                word_vector[1] = word
                sentence_matrix.append(word_vector)

            entity_matrix = []

            for e in entities:
                entity_vector = [None] * 4
                entity_vector[0] = str(e[0])
                entity_vector[1] = str(e[0])
                entity_vector[2] = e[1]
                entity_vector[3] = "1.0"

                entity_matrix.append(entity_vector)

            target_matrix = [[t,t] for t in targets]

            dataset.append((sentence_matrix, entity_matrix, target_matrix))

        return dataset

t = ToyDatasetMaker()

stop = False

while not stop:
    print("Input sentence:")
    from_console = input()

    if from_console.strip() == "stop":
        stop = True
        continue

    if from_console.strip() == "stats":
        dataset = t.get_dataset_representation()
        edges = np.array(t.global_edges)

        total_entities = np.unique(np.concatenate((edges[:, 0], edges[:, 2]))).shape[0]
        total_relations = np.unique(edges[:, 1]).shape[0]

        print("Total questions: " + str(len(dataset)))
        print("Total unique vertices: " + str(total_entities))
        print("Total unique relation types: " + str(total_relations))
        continue

    t.process(from_console)

dataset = t.get_dataset_representation()
edges = np.array(t.global_edges)

random.shuffle(dataset)
split = int(len(dataset)*0.8)

training_data = dataset[:split]
test_data = dataset[split:]

write_mode = "a+" if args.append else "w+"
graph_file = open(args.graph, write_mode)
train_file = open(args.train_file, write_mode)
test_file = open(args.test_file, write_mode)

for edge in edges:
    print(",".join(edge), file=graph_file)

def print_to_file(data, file):
    first = True
    for s_matrix, e_matrix, t_matrix in data:
        if first:
            first = False
        else:
            print("", file=file)

        print("\n".join(["\t".join(line) for line in s_matrix]), file=file)
        print("", file=file)
        print("\n".join(["\t".join(line) for line in e_matrix]), file=file)
        print("", file=file)
        print("\n".join(["\t".join(line) for line in t_matrix]), file=file)


print_to_file(training_data, train_file)
print_to_file(test_data, test_file)

print("I made a dataset!")

total_entities = np.unique(np.concatenate((edges[:,0], edges[:,2]))).shape[0]
total_relations = np.unique(edges[:,1]).shape[0]

print("Total unique vertices: " + str(total_entities))
print("Total unique relation types: " + str(total_relations))
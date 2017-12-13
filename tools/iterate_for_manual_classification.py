from helpers.read_conll_files import ConllReader
import random

reader = ConllReader("/home/michael/Projects/QuestionAnswering/GCNQA/data/webquestions/train.internal.conll")
sentences = []
for line in reader.iterate():
    sentence = " ".join([w[1] for w in line["sentence"]])
    sentences.append(sentence)

subset = random.sample(sentences, 50)

counter = {}

for example in subset:
    print(example)
    classification = input()

    if classification not in counter:
        counter[classification] = 0

    counter[classification] += 1

total = sum([v for k,v in counter.items()])
print("class\tcount\t%")
for k,v in counter.items():
    print(str(k) + ":   \t" + str(v) + "   \t" + str(v/total*100))
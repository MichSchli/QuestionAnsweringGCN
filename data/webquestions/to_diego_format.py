filename = "train.split.conll.uppercased"
outfile = "train.split.diego.uppercased"
outfile = open(outfile, "w")
with open(filename) as data_file:
    sentence_matrix = []

    reading_sentence = True
    reading_entities = False
    for line in data_file:
        line = line.strip()

        if line and reading_sentence:
            print(line, file=outfile)
        elif not line and reading_sentence:
            print("", file=outfile)
            reading_sentence = False
            reading_entities = True
        elif not line and reading_entities:
            reading_entities = False
        elif not line and not reading_sentence and not reading_entities:
            reading_sentence = True
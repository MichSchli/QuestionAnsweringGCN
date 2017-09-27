def iterate_entity_pairs():
    filename = "train.internal.conll"
    with open(filename) as data_file:
        sentence_matrix = []
        gold_matrix = []
        entity_matrix = []

        reading_sentence = True
        reading_entities = False
        for line in data_file:
            line = line.strip()

            if line and reading_sentence:
                sentence_matrix.append(line.split('\t'))
            elif line and reading_entities:
                entity_matrix.append(line.split('\t'))
            elif line and not reading_sentence and not reading_entities:
                gold_matrix.append(line.split('\t'))
            elif not line and reading_sentence:
                reading_sentence = False
                reading_entities = True
            elif not line and reading_entities:
                reading_entities = False
            elif not line and not reading_sentence and not reading_entities:
                reading_sentence = True

                yield [e[2] for e in entity_matrix], [g[1] for g in gold_matrix]


for e_list, g_list in iterate_entity_pairs():
    for g in g_list:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/>\n\nselect * where {\n?s ?r ?o .\n?o ?r2 \'"+g+"\'@en .\n"
        query += "values ?s { "+" ".join(["ns:"+e for e in e_list])+" }\n"
        query += "}"
        print(query)

    print(e_list)
    print(g_list)
    exit()
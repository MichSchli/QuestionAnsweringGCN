edge_count_file = open("edge_count.txt", "r")

count_dict = {}

for line in edge_count_file:
    parts = [p.strip() for p in line.strip().split(" ")]
    if parts[1].startswith("http://rdf.freebase.com/ns/"):
        count_dict[parts[1]] = parts[0]


old_schema_file = open("relation_schema.txt", "r")

new_count_dict = {}

for line in old_schema_file:
    line = line.strip()
    parts = line.strip().split("\t")
    if len(parts) > 1:
        relation = "http://rdf.freebase.com/ns/" + parts[0]
        if relation in count_dict:
            new_count_dict[relation] = count_dict[relation]

        if len(parts) > 3:
            relation = "http://rdf.freebase.com/ns/" + parts[3]
            if relation in count_dict:
                new_count_dict[relation] = count_dict[relation]

l = []
for relation,count in sorted(list(new_count_dict.items()), reverse=True, key=lambda x: int(x[1])):
    l.append(relation + "\t" + count)

print("\n".join(l))
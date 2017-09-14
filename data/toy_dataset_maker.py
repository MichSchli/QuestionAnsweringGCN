import argparse

parser = argparse.ArgumentParser(description='Creates a dataset from a toy graph.')
parser.add_argument('--file', type=str, help='The location of the .graph-file to be parsed')
args = parser.parse_args()

edges = []

stop = False

entity_vertices = []
event_vertices = []
literal_vertices = []

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

    if pattern == "1":
        s = words[0]
        v = words[1]
        o = words[2]

        s_type = "entity" if s.endswith("_") else "literal"
        o_type = "entity" if o.endswith("_") else "literal"

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

print(entity_vertices)
print(literal_vertices)
print(edges)

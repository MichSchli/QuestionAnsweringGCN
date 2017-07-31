from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface

iface = FreebaseInterface()
results = iface.retrieve_one_neighborhood_graph(["ns:m.014zcr", "ns:m.0q0b4"], limit=3000)
for result in results:
    print(result)
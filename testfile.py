from KnowledgeBaseInterface.FreebaseInterface import FreebaseInterface

iface = FreebaseInterface()
results = iface.get_neighborhood(["m.014zcr", "m.0q0b4"], edge_limit=4000, hops=2)
print(results[0].shape)
print(results[1].shape)

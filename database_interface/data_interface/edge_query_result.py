class EdgeQueryResult:

    edges = None
    vertices = None

    def __init__(self):
        self.edges = []
        self.vertices = {}

    def append_edge(self, edge):
        self.edges.append(edge)

    def append_vertex(self, vertex, type):
        self.vertices[vertex] = type
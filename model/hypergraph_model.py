from time import sleep

import numpy
import numpy as np

from model.vertex_feature_model import VertexFeatureModel


class HypergraphModel:

    event_vertices = None
    entity_vertices = None

    expandable_event_vertices = None
    expandable_entity_vertices = None

    event_to_entity_edges = None
    entity_to_event_edges = None
    entity_to_entity_edges = None

    discovered_entities = None
    discovered_events = None

    entity_map = None
    inverse_entity_map = None
    name_edge_type = -1
    type_edge_type = -1

    centroids = None

    name_map = None

    def to_index(self, entity):
        #print(self.inverse_entity_map)
        return self.inverse_entity_map[entity]

    def has_index(self, entity):
        return entity in self.inverse_entity_map

    def from_index(self, index):
        return self.entity_map[index]

    def __init__(self):
        self.event_vertices = np.empty(0)
        self.entity_vertices = np.empty(0)
        self.expandable_event_vertices = np.empty(0)
        self.expandable_entity_vertices = np.empty(0)
        self.expanded_event_vertices = np.empty(0)
        self.expanded_entity_vertices = np.empty(0)
        self.event_to_entity_edges = np.empty((0,3))
        self.entity_to_event_edges = np.empty((0,3))
        self.entity_to_entity_edges = np.empty((0,3))

        self.discovered_entities = np.empty(0)
        self.discovered_events = np.empty(0)

    def make_name_map(self):
        non_name_vertices = {}
        non_name_edges = []
        name_vertices = {}

        non_name_counter = 0
        name_counter = 0

        self.name_map = VertexFeatureModel()
        name_dict = {}

        for edge in self.entity_to_entity_edges:
            if edge[1] == self.name_edge_type:
                if edge[0] in name_vertices:
                    print("FOUND PROBLEM")
                    print(edge)
                    exit()

                if edge[0] not in non_name_vertices:
                    non_name_vertices[edge[0]] = non_name_counter
                    non_name_counter += 1

                if edge[2] not in name_vertices:
                    name_vertices[edge[2]] = name_counter
                    name_counter += 1

                name_dict[non_name_vertices[edge[0]]] = name_vertices[edge[2]]
            else:
                if edge[0] not in non_name_vertices:
                    non_name_vertices[edge[0]] = non_name_counter
                    non_name_counter += 1

                if edge[2] not in non_name_vertices:
                    non_name_vertices[edge[2]] = non_name_counter
                    non_name_counter += 1

                non_name_edges.append([non_name_vertices[edge[0]], edge[1], non_name_vertices[edge[2]]])

        for i,edge in enumerate(self.entity_to_event_edges):
            if edge[0] not in non_name_vertices:
                non_name_vertices[edge[0]] = non_name_counter
                non_name_counter += 1
            self.entity_to_event_edges[i][0] = non_name_vertices[edge[0]]

        for i,edge in enumerate(self.event_to_entity_edges):
            if edge[2] not in non_name_vertices:
                non_name_vertices[edge[2]] = non_name_counter
                non_name_counter += 1
            self.event_to_entity_edges[i][2] = non_name_vertices[edge[2]]

        for c in self.centroids:
            if c not in non_name_vertices:
                non_name_vertices[c] = non_name_counter
                non_name_counter += 1

        new_entity_map = {non_name_vertices[k]:v for k,v in self.entity_map.items() if k in non_name_vertices}
        new_inverse_entity_map = {k:non_name_vertices[v] for k,v in self.inverse_entity_map.items() if v in non_name_vertices}
        name_map_map = {self.entity_map[k]:v for k,v in name_vertices.items()}

        self.name_map.set_map(np.array(sorted(name_map_map.keys(), key=lambda k: name_map_map[k])), name_dict)
        self.entity_vertices = np.array(sorted(non_name_vertices.keys(), key=lambda k: non_name_vertices[k]))
        self.entity_to_entity_edges = np.array(non_name_edges) if len(non_name_edges) > 0 else np.empty((0,3), dtype=np.int32)

        #print(self.centroids)
        #print([c in non_name_vertices for c in self.centroids])
        #print([c in name_vertices for c in self.centroids])
        self.centroids = np.array([non_name_vertices[c] for c in self.centroids])

        self.entity_map = new_entity_map
        self.inverse_entity_map = new_inverse_entity_map

    def make_type_map(self):
        self.type_map = VertexFeatureModel()
        type_dict = {}

        for edge in self.entity_to_entity_edges:
            if edge[1] == self.type_edge_type:
                type_dict[edge[0]] = edge[2]

        self.name_map.set_map(np.array(type_dict.values()))


    def get_name_connections(self, entities):
        return self.name_map.project(entities)
        print(entities)
        name_dict = {k:i for i,k in enumerate(entities)}
        names = np.array(entities)
        
        for edge in self.entity_to_entity_edges:
            #if entities[0].startswith("http") and edge[1] == "http://www.w3.org/2000/01/rdf-schema#label":
            #    print(edge)
            #    sleep(0.5)
            if edge[1] == self.name_edge_type and edge[0] in name_dict:
                #print(edge)
                names[name_dict[edge[0]]] = edge[2]
        #exit()
        #print("names")
        #print(names)
        return names #{n[0]:np.unique(n[1]) for n in names.items()}

    def get_inverse_name_connections(self, names):
        return self.name_map.inverse_project(names)
        vertices = {name: [] for name in names}
        for edge in self.entity_to_entity_edges:
            if edge[1] == self.name_edge_type and edge[2] in names:
                vertices[edge[2]].append(edge[0])
        #print("vertices")
        #print(vertices)
        vertices = {n[0]:np.unique(n[1]) for n in vertices.items()}
        #print(vertices)
        return vertices


    def get_edges_and_hyperedges(self, start_vertex):
        # Entities
        paths = []
        start_to_entity_edges = np.where(self.entity_to_entity_edges[:,0] == start_vertex)
        start_to_entity_edges = self.entity_to_entity_edges[start_to_entity_edges]

        entity_to_start_edges = np.where(self.entity_to_entity_edges[:,2] == start_vertex)
        entity_to_start_edges = self.entity_to_entity_edges[entity_to_start_edges]
        inversed = np.empty_like(entity_to_start_edges)
        inversed[:,0] = entity_to_start_edges[:,2]
        inversed[:,2] = entity_to_start_edges[:,0]
        inversed[:,1] = numpy.core.defchararray.add(entity_to_start_edges[:,1],".inverse")
        start_to_entity_edges = np.concatenate((start_to_entity_edges, inversed))

        for edge in start_to_entity_edges:
            paths.append([list(edge)])

        # Events

        start_to_event_edges = np.where(self.entity_to_event_edges[:, 0] == start_vertex)
        start_to_event_edges = self.entity_to_event_edges[start_to_event_edges]

        event_to_start_edges = np.where(self.event_to_entity_edges[:, 2] == start_vertex)
        event_to_start_edges = self.event_to_entity_edges[event_to_start_edges]

        if event_to_start_edges.shape[0] > 0:
            inversed = np.empty_like(event_to_start_edges)
            inversed[:, 0] = event_to_start_edges[:, 2]
            inversed[:, 2] = event_to_start_edges[:, 0]
            inversed[:, 1] = numpy.core.defchararray.add(event_to_start_edges[:, 1], ".inverse")
            start_to_event_edges = np.concatenate((start_to_event_edges, inversed))

        intermediary_event_vertices = start_to_event_edges[:,2]
        intermediary_to_entity_edges = np.isin(self.event_to_entity_edges[:, 0], intermediary_event_vertices)
        intermediary_to_entity_edges = self.event_to_entity_edges[intermediary_to_entity_edges]

        entity_intermediary_to_edges = np.isin(self.entity_to_event_edges[:, 2], intermediary_event_vertices)
        entity_intermediary_to_edges = self.entity_to_event_edges[entity_intermediary_to_edges]
        if entity_intermediary_to_edges.shape[0] > 0:
            inversed = np.empty_like(entity_intermediary_to_edges)
            inversed[:, 0] = entity_intermediary_to_edges[:, 2]
            inversed[:, 2] = entity_intermediary_to_edges[:, 0]
            inversed[:, 1] = numpy.core.defchararray.add(entity_intermediary_to_edges[:, 1], ".inverse")
            intermediary_to_entity_edges = np.concatenate((intermediary_to_entity_edges, inversed))

        new_paths = []
        for edge_1 in start_to_event_edges:
            for edge_2 in intermediary_to_entity_edges:
                if edge_1[2] == edge_2[0]:
                    new_paths.append([list(edge_1), list(edge_2)])
        paths.extend(new_paths)

        return paths

    """
    Add vertices to the graph, guaranteeing uniqueness.
    """
    def add_vertices(self, vertices, type="entities"):
        previous = self.entity_vertices if type == "entities" else self.event_vertices

        vertices = np.unique(vertices)
        unique_vertices = vertices[np.isin(vertices, previous, invert=True)]

        if type == "entities":
            unique_vertices = unique_vertices[np.isin(unique_vertices, self.discovered_entities, invert=True)]
            self.discovered_entities = np.concatenate((self.discovered_entities, unique_vertices))
        else:
            unique_vertices = unique_vertices[np.isin(unique_vertices, self.discovered_events, invert=True)]
            self.discovered_events = np.concatenate((self.discovered_events, unique_vertices))

    def populate_discovered(self, type="entities"):
        if type == "entities":
            self.entity_vertices = np.concatenate((self.entity_vertices, self.discovered_entities))
            self.expandable_entity_vertices = np.concatenate((self.expandable_entity_vertices, self.discovered_entities))
            self.discovered_entities = np.empty(0)
        else:
            self.event_vertices = np.concatenate((self.event_vertices, self.discovered_events))
            self.expandable_event_vertices = np.concatenate((self.expandable_event_vertices, self.discovered_events))
            self.discovered_events = np.empty(0)

    def join_other_hypergraph(self, other):
        # New edges have at least one vertex in the old graph.
        new_entity_vertices = other.entity_vertices[np.isin(other.entity_vertices, self.entity_vertices, assume_unique=True, invert=True)]
        new_event_vertices = other.event_vertices[np.isin(other.event_vertices, self.event_vertices, assume_unique=True, invert=True)]

        self.entity_vertices = np.concatenate((self.entity_vertices, new_entity_vertices))
        self.event_vertices = np.concatenate((self.event_vertices, new_event_vertices))

        # Add edges with new subject entity
        # Add edges with new object entity
        new_entity_to_entity_edges = np.logical_or(
            np.isin(other.entity_to_entity_edges[:,0], new_entity_vertices),
            np.isin(other.entity_to_entity_edges[:,2], new_entity_vertices)
        )
        new_entity_to_entity_edges = other.entity_to_entity_edges[new_entity_to_entity_edges]

        new_event_to_entity_edges = np.logical_or(
            np.isin(other.event_to_entity_edges[:, 0], new_event_vertices),
            np.isin(other.event_to_entity_edges[:, 2], new_entity_vertices)
        )
        new_event_to_entity_edges = other.event_to_entity_edges[new_event_to_entity_edges]

        new_entity_to_event_edges = np.logical_or(
            np.isin(other.entity_to_event_edges[:, 0], new_entity_vertices),
            np.isin(other.entity_to_event_edges[:, 2], new_event_vertices)
        )
        new_entity_to_event_edges = other.entity_to_event_edges[new_entity_to_event_edges]

        self.entity_to_entity_edges = np.concatenate((self.entity_to_entity_edges, new_entity_to_entity_edges))
        self.entity_to_event_edges = np.concatenate((self.entity_to_event_edges, new_entity_to_event_edges))
        self.event_to_entity_edges = np.concatenate((self.event_to_entity_edges, new_event_to_entity_edges))

        self.centroids = np.unique(np.concatenate((self.centroids, other.centroids)))

    def update_edges(self, edges, sources="entities", targets="events"):
        if sources == "entities":
            if targets == "events":
                self.entity_to_event_edges = edges
            else:
                self.entity_to_entity_edges = edges
        else:
            self.event_to_entity_edges = edges

    def set_centroids(self, entities):
        self.centroids = np.array(entities)

    def update_vertices(self):
        #print("update")
        #print(np.max(self.entity_vertices))
        if self.centroids.shape[0] == 0:
            return

        visited_v = self.centroids
        visited_e = np.array([], dtype=np.int32)
        frontier = self.centroids

        while frontier.shape[0] > 0:
            outgoing_v = self.entity_to_entity_edges[np.isin(self.entity_to_entity_edges[:,0], frontier)][:,2]
            ingoing_v = self.entity_to_entity_edges[np.isin(self.entity_to_entity_edges[:,2], frontier)][:,0]

            outgoing_e = self.entity_to_event_edges[np.isin(self.entity_to_event_edges[:,0], frontier)][:,2]
            ingoing_e = self.event_to_entity_edges[np.isin(self.event_to_entity_edges[:,2], frontier)][:,0]

            all_e = np.unique(np.concatenate((outgoing_e, ingoing_e)))
            visited_e = np.unique(np.concatenate((visited_e, all_e)))

            ingoing_e_v = self.entity_to_event_edges[np.isin(self.entity_to_event_edges[:,2], all_e)][:,0]
            outgoing_e_v = self.event_to_entity_edges[np.isin(self.event_to_entity_edges[:,0], all_e)][:,2]

            all_v = np.unique(np.concatenate((outgoing_v, ingoing_v, outgoing_e_v, ingoing_e_v)))
            frontier = all_v[np.isin(all_v, visited_v, assume_unique=True, invert=True)]
            visited_v = np.concatenate((visited_v, frontier))

        self.entity_vertices = self.entity_vertices[visited_v]
        self.event_vertices = self.event_vertices[visited_e]

        self.entity_to_entity_edges = self.entity_to_entity_edges[np.logical_or(np.isin(self.entity_to_entity_edges[:,0], visited_v),
                                                                                np.isin(self.entity_to_entity_edges[:,2], visited_v))]

        self.entity_to_event_edges = self.entity_to_event_edges[np.logical_or(np.isin(self.entity_to_event_edges[:,0], visited_v),
                                                                                np.isin(self.entity_to_event_edges[:,2], visited_e))]

        self.event_to_entity_edges = self.event_to_entity_edges[np.logical_or(np.isin(self.event_to_entity_edges[:,0], visited_e),
                                                                              np.isin(self.event_to_entity_edges[:,2], visited_v))]



    """
    Get all seen vertices of a given type.
    """
    def get_vertices(self, type="entities", ignore_names=False):
        if type == "entities":
            if ignore_names and self.entity_to_entity_edges.shape[0] > 0:
                name_edges = self.entity_to_entity_edges[np.where(self.entity_to_entity_edges[:,1] == self.name_edge_type)]
                name_vertices = np.unique(name_edges[:,2])

                non_name_edges = self.entity_to_entity_edges[np.where(self.entity_to_entity_edges[:,1] != self.name_edge_type)]
                other_non_name_vertices = self.event_to_entity_edges[:,2]
                more_non_name_vertices = self.entity_to_event_edges[:,0]
                even_more_non_name_vertices = self.entity_to_entity_edges[:,0]
                non_name_vertices = np.unique(np.concatenate((non_name_edges[:,2], other_non_name_vertices, more_non_name_vertices, even_more_non_name_vertices)))
                name_vertices = name_vertices[np.isin(name_vertices, non_name_vertices, assume_unique=True, invert=True)]

                #print("heyo")
                #print(np.max(self.entity_vertices))
                if self.entity_vertices.shape[0] == 0:
                    print(self.entity_vertices)
                    print(self.entity_to_entity_edges)
                    exit()

                return self.entity_vertices[np.isin(self.entity_vertices, name_vertices, assume_unique=True, invert=True)]

            return self.entity_vertices
        else:
            return self.event_vertices

    def get_edges(self, sources="entities", targets="events", ignore_names=False):
        if sources == "entities" and targets == "events":
            return self.entity_to_event_edges
        elif sources == "events" and targets == "entities":
            return self.event_to_entity_edges
        elif sources == "entities" and targets == "entities":
            if ignore_names and self.entity_to_entity_edges.shape[0] > 0:
                return self.entity_to_entity_edges[np.where(self.entity_to_entity_edges[:,1] != self.name_edge_type)]

            return self.entity_to_entity_edges

    """
    Get all expandable vertices of a given type.
    Allows "popping" where returned elements are removed from the list of expandable elements.
    """
    def get_expandable_vertices(self, type="entities", pop=False):
        if type == "entities":
            elements = self.expandable_entity_vertices
            if pop:
                self.expandable_entity_vertices = np.empty(0)
        else:
            elements = self.expandable_event_vertices
            if pop:
                self.expandable_event_vertices = np.empty(0)

        return elements

    """
    Get all seen vertices of a given type.
    """
    def get_expanded_vertices(self, type="entities"):
        if type == "entities":
            return self.expanded_entity_vertices
        else:
            return self.expanded_event_vertices

    """
    Mark as expanded all vertices of a given type.
    """
    def mark_expanded(self, vertices, type="entities"):
        if type == "entities":
            self.expanded_entity_vertices = np.concatenate((self.expanded_entity_vertices, vertices))
        else:
            self.expanded_event_vertices = np.concatenate((self.expanded_event_vertices, vertices))


    """
    Expand a set of frontier vertices of a given type to all unseen vertices of a given type.

        - Remark: We always know source from where we are in the algorithm,
                  and we always know target from which freebase method we executed.

    """
    def expand(self, forward_edges, backward_edges, sources="entities", targets="events"):
        self.expand_forward(forward_edges, sources=sources, targets=targets)
        self.expand_backward(backward_edges, sources=sources, targets=targets)

    def add_discovered_vertices(self, forward_edges, backward_edges, type="entities"):
        if forward_edges.shape[0] > 0:
            self.add_vertices(forward_edges[:,2], type=type)

        if backward_edges.shape[0] > 0:
            self.add_vertices(backward_edges[:,0], type=type)


    """
    Expand a set of (source->target) edges from the sources. Allows edges within the frontier.
    """
    def expand_forward(self, edges, sources="entities", targets="events"):
        if not edges.shape[0] > 0:
            return

        target = edges[:,2]
        unseen_or_frontier_targets = np.isin(target, self.get_expanded_vertices(targets), invert=True)

        self.append_edges(edges[unseen_or_frontier_targets], sources=sources, targets=targets)

    """
    Expand a set of (target->source) edges from the sources. Disallows edges within the frontier.
    """
    def expand_backward(self, edges, sources="entities", targets="events"):
        if not edges.shape[0] > 0:
            return

        target = edges[:, 0]
        unseen_targets = np.isin(target, self.get_vertices(targets), invert=True)

        self.append_edges(edges[unseen_targets], sources=targets, targets=sources)


    """
    Append edges of a particular kind
    """
    def append_edges(self, edges, sources="entities", targets="events"):
        if sources == "entities" and targets == "events":
            self.entity_to_event_edges = np.concatenate((self.entity_to_event_edges, edges))
        elif sources == "events" and targets == "entities":
            self.event_to_entity_edges = np.concatenate((self.event_to_entity_edges, edges))
        elif sources == "entities" and targets == "entities":
            self.entity_to_entity_edges = np.concatenate((self.entity_to_entity_edges, edges))

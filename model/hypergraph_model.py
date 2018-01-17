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

    event_centroid_map = None

    def set_scores_to_zero(self):
        self.vertex_scores = np.zeros_like(self.entity_vertices, dtype=np.float32)
        self.event_scores = np.zeros_like(self.event_vertices, dtype=np.float32)

    def propagate_scores(self, centroid_scores):
        self.vertex_scores = np.zeros_like(self.entity_vertices, dtype=np.float32)
        self.event_scores = np.zeros_like(self.event_vertices, dtype=np.float32)

        centroid_score_map = {centroid: score for centroid, score in zip(self.centroids, centroid_scores)}

        for centroid in self.centroids:
            self.vertex_scores[centroid] = centroid_score_map[centroid]

        for edge in self.entity_to_entity_edges:
            if edge[0] in self.centroids:
                self.vertex_scores[edge[2]] = max(self.vertex_scores[edge[0]], self.vertex_scores[edge[2]])
            elif edge[2] in self.centroids:
                self.vertex_scores[edge[0]] = max(self.vertex_scores[edge[2]], self.vertex_scores[edge[0]])

        for edge in self.entity_to_event_edges:
            self.event_scores[edge[2]] = max(self.event_scores[edge[2]], self.vertex_scores[edge[0]])

        for edge in self.event_to_entity_edges:
            self.event_scores[edge[0]] = max(self.event_scores[edge[0]], self.vertex_scores[edge[2]])

        for edge in self.entity_to_event_edges:
            self.vertex_scores[edge[0]] = max(self.vertex_scores[edge[0]], self.event_scores[edge[2]])

        for edge in self.event_to_entity_edges:
            self.vertex_scores[edge[2]] = max(self.vertex_scores[edge[2]], self.event_scores[edge[0]])

    def get_split_graph(self):
        new_graph = HypergraphModel()
        new_graph.name_edge_type = self.name_edge_type
        new_graph.type_edge_type = self.type_edge_type
        new_graph.relation_map = self.relation_map

        new_graph.entity_to_entity_edges = []
        new_graph.entity_to_event_edges = []
        new_graph.event_to_entity_edges = []
        new_graph.centroids = []
        v_counter = 0
        e_counter = 0
        new_graph.entity_map = {}
        new_graph.inverse_entity_map = {}

        new_graph.entity_vertices = []
        new_graph.event_vertices = []

        for centroid in self.centroids:
            in_centroid_entity_map = {centroid: v_counter}
            new_graph.centroids.append(v_counter)
            new_graph.entity_vertices.append(self.entity_vertices[centroid])
            v_counter += 1
            for edge in self.entity_to_entity_edges:
                if edge[0] == centroid or edge[2] == centroid:
                    if edge[0] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[0]] = v_counter
                        new_graph.entity_vertices.append(self.entity_vertices[edge[0]])
                        v_counter += 1
                    if edge[2] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[2]] = v_counter
                        new_graph.entity_vertices.append(self.entity_vertices[edge[2]])
                        v_counter += 1
                    new_graph.entity_to_entity_edges.append([in_centroid_entity_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            in_centroid_event_map = {}
            for edge in self.entity_to_event_edges:
                if edge[0] == centroid:
                    if edge[2] not in in_centroid_event_map:
                        in_centroid_event_map[edge[2]] = e_counter
                        new_graph.event_vertices.append(self.event_vertices[edge[2]])
                        e_counter += 1
                    new_graph.entity_to_event_edges.append(
                        [in_centroid_entity_map[edge[0]], edge[1], in_centroid_event_map[edge[2]]])

            for edge in self.event_to_entity_edges:
                if edge[2] == centroid:
                    if edge[0] not in in_centroid_event_map:
                        in_centroid_event_map[edge[0]] = e_counter
                        new_graph.event_vertices.append(self.event_vertices[edge[0]])
                        e_counter += 1
                    new_graph.event_to_entity_edges.append(
                        [in_centroid_event_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            for edge in self.entity_to_event_edges:
                if edge[0] != centroid and edge[2] in in_centroid_event_map:
                    if edge[0] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[0]] = v_counter
                        new_graph.entity_vertices.append(self.entity_vertices[edge[0]])
                        v_counter += 1
                    new_graph.entity_to_event_edges.append(
                        [in_centroid_entity_map[edge[0]], edge[1], in_centroid_event_map[edge[2]]])

            for edge in self.event_to_entity_edges:
                if edge[2] != centroid and edge[0] in in_centroid_event_map:
                    if edge[2] not in in_centroid_entity_map:
                        in_centroid_entity_map[edge[2]] = v_counter
                        new_graph.entity_vertices.append(self.entity_vertices[edge[2]])
                        v_counter += 1
                    new_graph.event_to_entity_edges.append(
                        [in_centroid_event_map[edge[0]], edge[1], in_centroid_entity_map[edge[2]]])

            names = {}
            for k,v in in_centroid_entity_map.items():
                new_graph.entity_map[v] = self.entity_map[k]
                if self.entity_map[k] not in new_graph.inverse_entity_map:
                    new_graph.inverse_entity_map[self.entity_map[k]] = []
                new_graph.inverse_entity_map[self.entity_map[k]].append(v)

                if self.has_name(k):
                    names[v] = self.get_name(k)

            new_graph.add_names(names)

        new_graph.centroids = np.array(new_graph.centroids, dtype=np.int32)
        new_graph.entity_vertices = np.array(new_graph.entity_vertices, dtype=np.int32)
        new_graph.event_vertices = np.array(new_graph.event_vertices, dtype=np.int32)

        if len(new_graph.entity_to_entity_edges) > 0:
            new_graph.entity_to_entity_edges = np.array(new_graph.entity_to_entity_edges, dtype=np.int32)
        else:
            new_graph.entity_to_entity_edges = np.empty((0,3), dtype=np.int32)

        if len(new_graph.entity_to_event_edges) > 0:
            new_graph.entity_to_event_edges = np.array(new_graph.entity_to_event_edges, dtype=np.int32)
        else:
            new_graph.entity_to_event_edges = np.empty((0,3), dtype=np.int32)

        if len(new_graph.event_to_entity_edges) > 0:
            new_graph.event_to_entity_edges = np.array(new_graph.event_to_entity_edges, dtype=np.int32)
        else:
            new_graph.event_to_entity_edges = np.empty((0,3), dtype=np.int32)

        return new_graph

    def get_paths_to_neighboring_centroid(self, entity):
        l = []

        frontier = np.array([entity])

        outgoing_v = self.entity_to_entity_edges[np.logical_and(np.isin(self.entity_to_entity_edges[:, 0], frontier),
                                                                np.isin(self.entity_to_entity_edges[:, 2], self.centroids))]
        ingoing_v = self.entity_to_entity_edges[np.logical_and(np.isin(self.entity_to_entity_edges[:, 2], frontier),
                                                                np.isin(self.entity_to_entity_edges[:, 0], self.centroids))]

        for edge in outgoing_v:
            l.append([self.from_index_with_names(edge[0]) + " " + str(self.vertex_scores[edge[0]]), "->", self.relation_map[edge[1]], self.from_index_with_names(edge[2])])

        for edge in ingoing_v:
            l.append([self.from_index_with_names(edge[2]) + " " + str(self.vertex_scores[edge[2]]), "<-", self.relation_map[edge[1]], self.from_index_with_names(edge[0])])

        if self.event_centroid_map is None:
            self.event_centroid_map = {}
            for idx, edge in enumerate(self.event_to_entity_edges):
                if edge[2] in self.centroids:
                    if edge[0] not in self.event_centroid_map:
                        self.event_centroid_map[edge[0]] = []
                    self.event_centroid_map[edge[0]].append(["->", self.relation_map[edge[1]], idx, self.from_index_with_names(edge[2])])

            for edge in self.entity_to_event_edges:
                if edge[0] in self.centroids:
                    if edge[2] not in self.event_centroid_map:
                        self.event_centroid_map[edge[2]] = []
                    self.event_centroid_map[edge[2]].append(["<-", self.relation_map[edge[1]], self.from_index_with_names(edge[0])])

        for edge in self.entity_to_event_edges[np.isin(self.entity_to_event_edges[:, 0], frontier)]:
            if edge[2] in self.event_centroid_map:
                for e in self.event_centroid_map[edge[2]]:
                    representation = [self.from_index_with_names(edge[0]) + " " + str(self.vertex_scores[edge[0]]), " ->", self.relation_map[edge[1]], "e"+str(edge[2])]
                    representation.extend(e)
                    l.append(representation)

        for edge in self.event_to_entity_edges[np.isin(self.event_to_entity_edges[:, 2], frontier)]:
            if edge[0] in self.event_centroid_map:
                for e in self.event_centroid_map[edge[0]]:
                    representation = [self.from_index_with_names(edge[2]) + " " + str(self.vertex_scores[edge[2]]), " <-", self.relation_map[edge[1]], "e"+str(edge[0])]
                    representation.extend(e)
                    l.append(representation)

        return l

    def get_paths_to_neighboring_centroid_formal_todo_rename(self, target_entities):
        known_events = []
        entity_to_entity = []
        event_to_entity = []
        entity_to_event = []

        outgoing_v = np.logical_and(np.isin(self.entity_to_entity_edges[:, 0], target_entities),
                                            np.isin(self.entity_to_entity_edges[:, 2], self.centroids))
        ingoing_v = np.logical_and(np.isin(self.entity_to_entity_edges[:, 2], target_entities),
                                            np.isin(self.entity_to_entity_edges[:, 0], self.centroids))

        en_to_en_keep = np.logical_or(ingoing_v, outgoing_v)

        has_centroid_connection = np.zeros_like(self.event_vertices, dtype=np.bool)
        has_multiple_centroid_connections = np.zeros_like(self.event_vertices, dtype=np.bool)
        has_kept_connection = np.zeros_like(self.event_vertices, dtype=np.bool)

        for edge in self.event_to_entity_edges:
            if edge[2] in self.centroids:
                if has_centroid_connection[edge[0]]:
                    has_multiple_centroid_connections[edge[0]]
                else:
                    has_centroid_connection[edge[0]] = True
            elif edge[2] in target_entities:
                has_kept_connection[edge[0]] = True

        for edge in self.entity_to_event_edges:
            if edge[0] in self.centroids:
                if has_centroid_connection[edge[2]]:
                    has_multiple_centroid_connections[edge[2]]
                else:
                    has_centroid_connection[edge[2]] = True
            elif edge[0] in target_entities:
                has_kept_connection[edge[2]] = True

        events_to_keep = np.logical_or(np.logical_and(has_centroid_connection, has_kept_connection),
                                       has_multiple_centroid_connections)
        events_to_keep = self.event_vertices[events_to_keep]


        en_to_ev_keep = np.logical_and(np.isin(self.entity_to_event_edges[:, 0], target_entities),
                                        np.isin(self.entity_to_event_edges[:, 2], events_to_keep))
        ev_to_en_keep = np.logical_and(np.isin(self.event_to_entity_edges[:, 0], events_to_keep),
                                        np.isin(self.event_to_entity_edges[:, 2], target_entities))

        entity_to_entity = self.entity_to_entity_edges[en_to_en_keep]
        entity_to_event = self.entity_to_event_edges[en_to_ev_keep]
        event_to_entity = self.event_to_entity_edges[ev_to_en_keep]

        return entity_to_entity, entity_to_event, event_to_entity, events_to_keep

    def from_index_with_names(self, index):
        if self.has_name(index):
            return self.get_name(index)
        else:
            return self.from_index(index)

    def to_index(self, entity):
        return self.inverse_entity_map[entity]

    def has_index(self, entity):
        return entity in self.inverse_entity_map

    def from_index(self, index):
        return self.entity_map[index] if index in self.entity_map else "UNKNOWN_VERTEX"

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

        self.name_map = VertexFeatureModel()

    def add_names(self, name_map):
        self.name_map.add_features(name_map)

    def has_name(self, entity):
        return self.name_map.has_projection(entity)

    def get_name(self, entity):
        return self.name_map.project_singleton(entity)

    def get_name_connections(self, entities):
        return self.name_map.project(entities)

    def get_inverse_name_connections(self, names):
        return self.name_map.inverse_project(names)

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
        # If source is events, frontier has already been expanded to nearby events.
        if sources == "events":
            unseen_or_frontier_targets = np.isin(target, self.get_vertices(targets), invert=True)
        else:
            unseen_or_frontier_targets = np.isin(target, self.get_expanded_vertices(targets), invert=True)

        #unseen_or_frontier_targets = np.isin(target, self.get_expanded_vertices(targets), invert=True)

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

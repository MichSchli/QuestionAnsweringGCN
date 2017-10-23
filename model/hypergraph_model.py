from time import sleep

import numpy as np


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

    """
    Cache storage:
    """

    def to_string_storage(self):
        l = ":::".join(self.event_vertices)
        l += "$$$" + ":::".join(self.entity_vertices)
        l += "$$$" + self.string_store_list(self.event_to_entity_edges)
        l += "$$$" + self.string_store_list(self.entity_to_event_edges)
        l += "$$$" + self.string_store_list(self.entity_to_entity_edges)

        return l

    def string_store_list(self, l):
        s = ""
        first = True
        for edge in l:
            if first:
                first = False
            else:
                s += ":::"
            s += "%%%".join(edge)

        return s

    def load_from_string_storage(self, string):
        event_vertices, counter = self.load_vertices_from_string(string, -1)
        entity_vertices, counter = self.load_vertices_from_string(string, counter)
        event_to_entity_edges, counter = self.load_edges_from_string(string, counter)
        entity_to_event_edges, counter = self.load_edges_from_string(string, counter)
        entity_to_entity_edges, counter = self.load_edges_from_string(string, counter)

        self.event_vertices = event_vertices
        self.entity_vertices = entity_vertices
        self.event_to_entity_edges = event_to_entity_edges
        self.entity_to_event_edges = entity_to_event_edges
        self.entity_to_entity_edges = entity_to_entity_edges

    def load_vertices_from_string(self, string, counter):
        vertices = []
        parsed_element = ""
        while counter < len(string)-1:
            counter += 1
            character = string[counter:counter+3]
            if character == ":::":
                counter += 2
                vertices.append(parsed_element)
                parsed_element = ""
            elif character == "$$$":
                counter += 2
                vertices.append(parsed_element)
                return np.array(vertices), counter
            else:
                parsed_element += character[0]

        return np.array(vertices), counter

    def load_edges_from_string(self, string, counter):
        edges = []
        parsed_element = [""]
        while counter < len(string)-1:
            counter += 1
            character = string[counter:counter+3]
            if character == ":::":
                counter += 2
                edges.append(parsed_element)
                parsed_element = [""]
            elif character == "%%%":
                counter += 2
                parsed_element.append("")
            elif character == "$$$":
                counter += 2
                if len(parsed_element) == 3:
                    edges.append(parsed_element)

                if len(edges) == 0:
                    return np.empty((0,3), dtype=np.int32), counter
                return np.array(edges), counter
            else:
                parsed_element[-1] += character[0]

        if len(parsed_element) == 3:
            edges.append(parsed_element)
        if len(edges) == 0:
            return np.empty((0,3), dtype=np.int32), counter

        return np.array(edges), counter

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

    """
    Get all seen vertices of a given type.
    """
    def get_vertices(self, type="entities"):
        if type == "entities":
            return self.entity_vertices
        else:
            return self.event_vertices

    def get_edges(self, sources="entities", targets="events"):
        if sources == "entities" and targets == "events":
            return self.entity_to_event_edges
        elif sources == "events" and targets == "entities":
            return self.event_to_entity_edges
        elif sources == "entities" and targets == "entities":
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
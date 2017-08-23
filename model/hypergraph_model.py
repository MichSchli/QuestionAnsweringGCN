import numpy as np


class HypergraphModel:

    event_vertices = None
    entity_vertices = None

    expandable_event_vertices = None
    expandable_entity_vertices = None

    event_to_entity_edges = None
    event_to_event_edges = None
    entity_to_event_edges = None
    entity_to_entity_edges = None

    def __init__(self):
        self.event_vertices = np.empty(0)
        self.entity_vertices = np.empty(0)
        self.expandable_event_vertices = np.empty(0)
        self.expandable_entity_vertices = np.empty(0)
        self.expanded_event_vertices = np.empty(0)
        self.expanded_entity_vertices = np.empty(0)
        self.event_to_entity_edges = np.empty((0,3))
        self.event_to_event_edges = np.empty((0,3))
        self.entity_to_event_edges = np.empty((0,3))
        self.entity_to_entity_edges = np.empty((0,3))

    """
    Add vertices to the graph, guaranteeing uniqueness.
    """
    def add_vertices(self, vertices, type="entities"):
        previous = self.entity_vertices if type == "entities" else self.event_vertices

        vertices = np.unique(vertices)
        unique_vertices = vertices[np.isin(vertices, previous, invert=True)]

        if type == "entities":
            self.entity_vertices = np.concatenate((previous, unique_vertices))
            self.expandable_entity_vertices = np.concatenate((self.expandable_entity_vertices, unique_vertices))
        else:
            self.event_vertices = np.concatenate((previous, unique_vertices))
            self.expandable_event_vertices = np.concatenate((self.expandable_event_vertices, unique_vertices))

    """
    Get all seen vertices of a given type.
    """
    def get_seen_vertices(self, type="entities"):
        if type == "entities":
            return self.entity_vertices
        else:
            return self.event_vertices

    """
    Get all expandable vertices of a given type.
    """
    def get_expandable_vertices(self, type="entities"):
        if type == "entities":
            return self.expandable_entity_vertices
        else:
            return self.expandable_event_vertices

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
            self.expandable_entity_vertices = np.empty(0)
            self.expanded_entity_vertices = np.concatenate((self.expanded_entity_vertices, vertices))
        else:
            self.expandable_event_vertices = np.empty(0)
            self.expanded_event_vertices = np.concatenate((self.expanded_event_vertices, vertices))


    """
    Expand a set of frontier vertices of a given type to all unseen vertices of a given type.

        - Remark: We always know source from where we are in the algorithm,
                  and we always know target from which freebase method we executed.

    """
    def expand(self, frontier, forward_edges, backward_edges, sources="entities", targets="events"):
        # We first expand in the forward direction:
        self.expand_forward(forward_edges, sources=sources, targets=targets)

        # Then we expand in the backward direction:
        self.expand_backward(backward_edges, sources=sources, targets=targets)

        # Then we add all new vertices and mark EXPANDABLE when appropriate:
        self.add_vertices(forward_edges[:,2], type=targets)
        self.add_vertices(backward_edges[:,0], type=targets)

    """
    Clear expandables, mark frontier expanded
    """
    def clear_expandable_vertices(self, frontier, type="entities"):
        # Finally we mark the frontier as expanded:
        self.mark_expanded(frontier, type=type)


    """
    Expand a set of (source->target) edges from the sources. Allows edges within the frontier.
    """
    def expand_forward(self, edges, sources="entities", targets="events"):
        target = edges[:,2]
        unseen_or_frontier_targets = np.isin(target, self.get_expanded_vertices(targets), invert=True)

        self.append_edges(edges[unseen_or_frontier_targets], sources=sources, targets=targets)

    """
    Expand a set of (target->source) edges from the sources. Disallows edges within the frontier.
    """
    def expand_backward(self, edges, sources="entities", targets="events"):
        target = edges[:, 0]
        unseen_targets = np.isin(target, self.get_seen_vertices(targets), invert=True)

        self.append_edges(edges[unseen_targets], sources=targets, targets=sources)


    """
    Append edges of a particular kind
    """
    def append_edges(self, edges, sources="entities", targets="events"):
        if sources == "entities" and targets == "events":
            self.entity_to_event_edges = np.concatenate((self.entity_to_event_edges, edges))
        elif sources == "events" and targets == "entities":
            self.entity_to_event_edges = np.concatenate((self.entity_to_event_edges, edges))
        elif sources == "entities" and targets == "entities":
            self.entity_to_entity_edges = np.concatenate((self.entity_to_entity_edges, edges))
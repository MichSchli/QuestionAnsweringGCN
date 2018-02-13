from tkinter import *
from PIL import Image
import argparse
import random
import tkinter as tk

parser = argparse.ArgumentParser(description='Graphically illustrates graphs from an analysis file.')
parser.add_argument('--file', type=str, help='The location of the analysis file to be shown')
args = parser.parse_args()

examples = [{"sentence": None,
             "entity_to_event_edges": [],
             "event_to_entity_edges": [],
             "entity_to_entity_edges": []}]
data_type = 0

for line in open(args.file):
    line = line.strip()

    if not line:
        examples.append({"sentence": None,
                         "entity_to_event_edges": [],
                         "event_to_entity_edges": [],
                         "entity_to_entity_edges": []})
        data_type = 0
    elif line == "-----":
        data_type += 1
    else:
        if data_type == 0:
            examples[-1]["sentence"] = line
        elif data_type == 1:
            examples[-1]["entity_to_event_edges"].append(line.split("\t"))
        elif data_type == 2:
            examples[-1]["event_to_entity_edges"].append(line.split("\t"))
        elif data_type == 3:
            examples[-1]["entity_to_entity_edges"].append(line.split("\t"))

class Graph:

    centroids = None
    golds = None

    def __init__(self, example):
        self.centroids = {}
        self.golds = {}
        self.find_vertices(example)

        self.entity_to_event_edges = example["entity_to_event_edges"]
        self.event_to_entity_edges = example["event_to_entity_edges"]
        self.entity_to_entity_edges = example["entity_to_entity_edges"]

    def get_centroids(self):
        centroids = []
        for label, vertex in self.vertices.items():
            if vertex["is_centroid"]:
                centroids.append([label, vertex["score"], vertex["connections"]])

        return centroids

    def get_other_vertices(self):
        vs = []
        for label, vertex in self.vertices.items():
            if not vertex["is_centroid"]:
                vs.append(label)

        return vs

    def get_events(self):
        return self.events

    def get_all_edges(self):
        all_edges = []
        all_edges.extend(self.entity_to_event_edges)
        all_edges.extend(self.event_to_entity_edges)
        all_edges.extend(self.entity_to_entity_edges)
        return all_edges

    def find_vertices(self, example):
        vertices = {}
        events = {}
        for edge in example["entity_to_event_edges"]:
            if edge[0] not in vertices:
                vertices[edge[0]] = {"is_gold" : False,
                                     "is_centroid": False,
                                     "score": None,
                                     "connections": 0}

                if edge[8] != "_":
                    score = float(edge[8].split("=")[-1])
                    vertices[edge[0]]["is_centroid"] = True
                    vertices[edge[0]]["score"] = score

                if edge[6] != "_":
                    vertices[edge[0]]["is_gold"] = True

            vertices[edge[0]]["connections"] += 1

            if edge[2] not in events:
                events[edge[2]] = True


        for edge in example["event_to_entity_edges"]:
            if edge[2] not in vertices:
                vertices[edge[2]] = {"is_gold" : False,
                                     "is_centroid": False,
                                     "score": None,
                                     "connections": 0}

                if edge[9] != "_":
                    score = float(edge[9].split("=")[-1])
                    vertices[edge[2]]["is_centroid"] = True
                    vertices[edge[2]]["score"] = score

                if edge[7] != "_":
                    vertices[edge[2]]["is_gold"] = True

            vertices[edge[2]]["connections"] += 1

            if edge[0] not in events:
                events[edge[0]] = True


        for edge in example["entity_to_entity_edges"]:
            if edge[0] not in vertices:
                vertices[edge[0]] = {"is_gold" : False,
                                     "is_centroid": False,
                                     "score": None,
                                     "connections": 0}

                if edge[8] != "_":
                    score = float(edge[8].split("=")[-1])
                    vertices[edge[0]]["is_centroid"] = True
                    vertices[edge[0]]["score"] = score

            if edge[2] not in vertices:
                vertices[edge[2]] = {"is_gold": None,
                                     "is_centroid": None,
                                     "score": None,
                                     "connections": 0}

                if edge[9] != "_":
                    score = float(edge[9].split("=")[-1])
                    vertices[edge[2]]["is_centroid"] = True
                    vertices[edge[2]]["score"] = score

                if edge[7] != "_":
                   vertices[edge[2]]["is_gold"] = True

            vertices[edge[0]]["connections"] += 1
            vertices[edge[2]]["connections"] += 1

        self.vertices = vertices
        self.events = events





root = Tk()
frame = Frame(root)

canvas_width = 1024
canvas_height = 720

cv = Canvas(frame, width=canvas_width, height=canvas_height, bg='white')
cv.grid(row=0, columnspan=4)


def save_button():
    cv.postscript(file="graph_cap.eps")
    img = Image.open("graph_cap.eps")
    img.save("graph_cap.png", "png")
    print("done")

save = Button(frame, text="save", fg="black", command=save_button)
save.grid(row=1, column=2)


def forward_button():
    global example_pointer
    if example_pointer == len(examples) -1:
        return
    example_pointer += 1
    draw_example(example_pointer)

edge_objects = None

location_map = {}
graph = None

def draw_example(example_pointer):
    global location_map
    global graph
    cv.delete("all")
    cv.create_text(canvas_width / 2, canvas_height - 50, text=examples[example_pointer]["sentence"], anchor="s",
                   fill="black", font=("Helvetica", 24))
    graph = Graph(examples[example_pointer])
    centroids = graph.get_centroids()
    padding = 50
    bottom_padding = 200
    text_height = 30
    vertex_radius = 5
    draw_width = canvas_width - padding*2
    centroid_locations = [draw_width / len(centroids) * (i+0.5) + padding for i in range(len(centroids))]

    location_map = {}

    for centroid, location in zip(centroids, centroid_locations):
        vertical = canvas_height - bottom_padding - text_height*2 - 10
        cv.create_oval(location-vertex_radius,
                       vertical - vertex_radius,
                       location+vertex_radius,
                       vertical + vertex_radius)
        cv.create_text(location, canvas_height - bottom_padding - text_height, text=centroid[0], anchor="s",
                   fill="black", font=("Helvetica", 12))

        cv.create_text(location, canvas_height - bottom_padding, text=centroid[1], anchor="s",
                   fill="black", font=("Helvetica", 12))

        location_map[centroid[0]] = [location, vertical]

    forest_border = 30

    forest_top = forest_border
    forest_bottom = canvas_height - forest_border - bottom_padding - 2*text_height - 10
    forest_left = forest_border
    forest_right = canvas_width - forest_border

    for vertex in graph.get_other_vertices():
        vertical = random.randint(forest_top, forest_bottom)
        horizontal = random.randint(forest_left, forest_right)

        cv.create_oval(horizontal - vertex_radius,
                       vertical - vertex_radius,
                       horizontal + vertex_radius,
                       vertical + vertex_radius)
        cv.create_text(horizontal, vertical + text_height, text=vertex, anchor="s",
                       fill="black", font=("Helvetica", 12))

        if vertex not in location_map:
            location_map[vertex] = [horizontal, vertical]

    cvt_radius = 2
    cvt_extra_border = 40

    for vertex in graph.get_events():
        vertical = random.randint(forest_top+cvt_extra_border, forest_bottom-cvt_extra_border)
        horizontal = random.randint(forest_left+cvt_extra_border, forest_right-cvt_extra_border)

        cv.create_rectangle(horizontal - cvt_radius,
                            vertical - cvt_radius,
                            horizontal + cvt_radius,
                            vertical + cvt_radius)
        #cv.create_text(horizontal, vertical + text_height, text=vertex, anchor="s",
        #               fill="black", font=("Helvetica", 12))

        if vertex not in location_map:
            location_map[vertex] = [horizontal, vertical]

    gate_cutoff = 0.0

    draw_edges(gate_cutoff, graph, location_map)


def draw_edges(gate_cutoff, graph, location_map):
    global edge_objects

    if edge_objects is not None:
        for e in edge_objects:
            cv.delete(e)

    edge_objects = []
    for edge in graph.get_all_edges():

        forward_gate = float(edge[3])
        backward_gate = float(edge[4])

        if forward_gate < gate_cutoff and backward_gate < gate_cutoff:
            continue
        elif forward_gate > gate_cutoff and backward_gate > gate_cutoff:
            arrows = tk.BOTH
        elif forward_gate > gate_cutoff:
            arrows = tk.LAST
        else:
            arrows = tk.FIRST

        location_subject = location_map[edge[0]]
        location_object = location_map[edge[2]]

        line = cv.create_line(location_subject[0], location_subject[1], location_object[0], location_object[1], arrow=arrows)

        middle_horizontal = int(location_subject[0] + location_object[0]) / 2
        middle_vertical = int(location_subject[1] + location_object[1]) / 2

        text = cv.create_text(middle_horizontal, middle_vertical + 4, text=edge[1], anchor="s",
                       fill="black", font=("Helvetica", 8))

        edge_objects.append(line)
        edge_objects.append(text)

    return edge_objects


example_pointer = 0
draw_example(example_pointer)


def backward_button():
    global example_pointer
    if example_pointer == 0:
        return
    example_pointer -= 1
    draw_example(example_pointer)

back = Button(frame, text="<", fg="black", command=backward_button)
back.grid(row=1, column=0)

forward = Button(frame, text=">", fg="black", command=forward_button)
forward.grid(row=1, column=3)



def use_slider(event):
    global location_map
    percentage = float(event) / 100
    draw_edges(percentage, graph, location_map)

w = Scale(frame, orient=HORIZONTAL, from_=0, to=100, command=use_slider)
w.grid(row=1, column=1)


frame.pack()
root.mainloop()
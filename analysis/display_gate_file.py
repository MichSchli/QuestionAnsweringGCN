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
        if len(examples) == 4:
            break
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
                centroids.append([label, vertex["score"], vertex["strongest_connection"], vertex["prediction_probability"], vertex["is_gold"]])

        return centroids

    def get_other_vertices(self):
        vs = []
        for label, vertex in self.vertices.items():
            if not vertex["is_centroid"]:
                vs.append((label, vertex["strongest_connection"], vertex["prediction_probability"], vertex["is_gold"]))

        return vs

    def get_events(self):
        return [(label, vertex["strongest_connection"]) for label,vertex in self.events.items()]

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
                                     "strongest_connection": [0 for _ in edge[3].split("/")],
                                     "prediction_probability": float(edge[10])}

                if edge[8] != "_":
                    score = float(edge[8].split("=")[-1])
                    vertices[edge[0]]["is_centroid"] = True
                    vertices[edge[0]]["score"] = score

                if edge[6] != "_":
                    vertices[edge[0]]["is_gold"] = True

            score_forward = [float(s) for s in edge[3].split("/")]
            score_backward = [float(s) for s in edge[4].split("/")]
            max_scores = [max(sf, sb) for sf, sb in zip(score_forward, score_backward)]
            vertices[edge[0]]["strongest_connection"] = [max(sf, sb) for sf, sb in zip(vertices[edge[0]]["strongest_connection"], max_scores)]

            if edge[2] not in events:
                events[edge[2]] = {"strongest_connection": [0 for _ in edge[3].split("/")]}
            events[edge[2]]["strongest_connection"] = [max(sf, sb) for sf, sb in zip(events[edge[2]]["strongest_connection"], max_scores)]


        for edge in example["event_to_entity_edges"]:
            if edge[2] not in vertices:
                vertices[edge[2]] = {"is_gold" : False,
                                     "is_centroid": False,
                                     "score": None,
                                     "strongest_connection": [0 for _ in edge[3].split("/")],
                                     "prediction_probability": float(edge[11])}

                if edge[9] != "_":
                    score = float(edge[9].split("=")[-1])
                    vertices[edge[2]]["is_centroid"] = True
                    vertices[edge[2]]["score"] = score

                if edge[7] != "_":
                    vertices[edge[2]]["is_gold"] = True

            print(edge)
            score_forward = [float(s) for s in edge[3].split("/")]
            score_backward = [float(s) for s in edge[4].split("/")]
            max_scores = [max(sf, sb) for sf, sb in zip(score_forward, score_backward)]
            vertices[edge[2]]["strongest_connection"] = [max(sf, sb) for sf, sb in
                                                         zip(vertices[edge[2]]["strongest_connection"], max_scores)]

            if edge[0] not in events:
                events[edge[0]] = {"strongest_connection": [0 for _ in edge[3].split("/")]}
            events[edge[0]]["strongest_connection"] = [max(sf, sb) for sf, sb in
                                                       zip(events[edge[0]]["strongest_connection"], max_scores)]

        for edge in example["entity_to_entity_edges"]:
            if edge[0] not in vertices:
                vertices[edge[0]] = {"is_gold" : False,
                                     "is_centroid": False,
                                     "score": None,
                                     "strongest_connection": [0 for _ in edge[3].split("/")],
                                     "prediction_probability": float(edge[10])}

                if edge[8] != "_":
                    score = float(edge[8].split("=")[-1])
                    vertices[edge[0]]["is_centroid"] = True
                    vertices[edge[0]]["score"] = score

            if edge[2] not in vertices:
                vertices[edge[2]] = {"is_gold": None,
                                     "is_centroid": None,
                                     "score": None,
                                     "strongest_connection": [0 for _ in edge[3].split("/")],
                                     "prediction_probability": float(edge[11])}

                if edge[9] != "_":
                    score = float(edge[9].split("=")[-1])
                    vertices[edge[2]]["is_centroid"] = True
                    vertices[edge[2]]["score"] = score

                if edge[7] != "_":
                   vertices[edge[2]]["is_gold"] = True

            score_forward = [float(s) for s in edge[3].split("/")]
            score_backward = [float(s) for s in edge[4].split("/")]
            max_scores = [max(sf, sb) for sf, sb in zip(score_forward, score_backward)]
            vertices[edge[0]]["strongest_connection"] = [max(sf, sb) for sf, sb in
                                                         zip(vertices[edge[0]]["strongest_connection"], max_scores)]
            vertices[edge[2]]["strongest_connection"] = [max(sf, sb) for sf, sb in
                                                         zip(vertices[edge[2]]["strongest_connection"], max_scores)]

        self.vertices = vertices
        self.events = events




class AnalysisTool:

    root = None
    canvas = None

    canvas_width = 1024
    canvas_height = 720

    example_pointer = None
    examples = None

    edge_objects = None

    apply_cutoff_to_vertices = True
    preserve_predictions = True
    preserve_gold = True
    cutoff_percentage = 0.0
    layer = 0

    def __init__(self, layers):
        self.example_pointer = 0

        self.root = Tk()
        frame = Frame(self.root)
        self.canvas = Canvas(frame, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.grid(row=0, columnspan=5)

        save = Button(frame, text="save", fg="black", command=self.save_button)
        save.grid(row=1, column=3)

        back = Button(frame, text="<", fg="black", command=self.backward_button)
        back.grid(row=1, column=0)

        forward = Button(frame, text=">", fg="black", command=self.forward_button)
        forward.grid(row=1, column=4)

        w = Scale(frame, orient=HORIZONTAL, from_=0, to=100, command=self.use_slider)
        w.grid(row=1, column=1)

        w_l = Scale(frame, orient=HORIZONTAL, from_=1, to=layers, command=self.use_layer_slider)
        w_l.grid(row=1, column=2)

        frame.pack()

    def read_examples(self, examples):
        self.examples = [(example["sentence"], Graph(example)) for example in examples]
        self.draw_example()

    def run(self):
        self.root.mainloop()

    def save_button(self):
        self.canvas.postscript(file="graph_cap.eps")
        img = Image.open("graph_cap.eps")
        img.save("graph_cap.png", "png")

    def forward_button(self):
        if self.example_pointer == len(self.examples) - 1:
            return
        self.example_pointer += 1
        self.draw_example()

    def backward_button(self):
        if self.example_pointer == 0:
            return
        self.example_pointer -= 1
        self.draw_example()

    def use_slider(self, event):
        self.cutoff_percentage = float(event) / 100

        if self.apply_cutoff_to_vertices:
            self.draw_example(maintain_locations=True)
        else:
            self.draw_edges()

    def use_layer_slider(self, event):
        self.layer = int(event) - 1

        if self.apply_cutoff_to_vertices:
            self.draw_example(maintain_locations=True)
        else:
            self.draw_edges()

    """
    Drawing methods:
    """

    def draw_example(self, maintain_locations=False):
        sentence = self.examples[self.example_pointer][0]
        graph = self.examples[self.example_pointer][1]

        self.canvas.delete("all")
        self.canvas.create_text(self.canvas_width / 2, self.canvas_height - 50, text=sentence, anchor="s",
                       fill="black", font=("Helvetica", 24))

        centroids = graph.get_centroids()
        padding = 50
        prediction_circle_extra_radius = 4
        bottom_padding = 200
        text_height = 30
        vertex_radius = 5
        draw_width = self.canvas_width - padding * 2
        centroid_locations = [draw_width / len(centroids) * (i + 0.5) + padding for i in range(len(centroids))]

        if not maintain_locations:
            self.current_location_map = {}

        for centroid, location in zip(centroids, centroid_locations):
            vertical = self.canvas_height - bottom_padding - text_height * 2 - 10
            prediction_probability = centroid[3]

            if prediction_probability > 0.5:
                self.canvas.create_oval(location - vertex_radius-prediction_circle_extra_radius,
                                        vertical - vertex_radius-prediction_circle_extra_radius,
                                        location + vertex_radius+prediction_circle_extra_radius,
                                        vertical + vertex_radius+prediction_circle_extra_radius,
                                        fill="green")

            color = "yellow" if centroid[4] else "red"

            self.canvas.create_oval(location - vertex_radius,
                           vertical - vertex_radius,
                           location + vertex_radius,
                           vertical + vertex_radius, fill=color)
            self.canvas.create_text(location, self.canvas_height - bottom_padding - text_height, text=centroid[0], anchor="s",
                           fill="black", font=("Helvetica", 12))

            self.canvas.create_text(location, self.canvas_height - bottom_padding, text=centroid[1], anchor="s",
                           fill="black", font=("Helvetica", 12))

            self.current_location_map[centroid[0]] = [location, vertical]

        forest_border = 30

        forest_top = forest_border
        forest_bottom = self.canvas_height - forest_border - bottom_padding - 2 * text_height - 10
        forest_left = forest_border
        forest_right = self.canvas_width - forest_border

        for vertex in graph.get_other_vertices():
            if self.apply_cutoff_to_vertices \
                    and vertex[1][self.layer] < self.cutoff_percentage\
                    and not (vertex[3] and self.preserve_gold)\
                    and not (vertex[2] > 0.5 and self.preserve_predictions):
                continue
            prediction_probability = vertex[2]
            color = "yellow" if vertex[3] else "black"
            vertex = vertex[0]

            if vertex in self.current_location_map:
                vertical = self.current_location_map[vertex][1]
                horizontal = self.current_location_map[vertex][0]
            else:
                vertical = random.randint(forest_top, forest_bottom)
                horizontal = random.randint(forest_left, forest_right)
                self.current_location_map[vertex] = [horizontal, vertical]

            if prediction_probability > 0.5:
                self.canvas.create_oval(horizontal - vertex_radius-prediction_circle_extra_radius,
                                        vertical - vertex_radius-prediction_circle_extra_radius,
                                        horizontal + vertex_radius+prediction_circle_extra_radius,
                                        vertical + vertex_radius+prediction_circle_extra_radius,
                                        fill="green")

            self.canvas.create_oval(horizontal - vertex_radius,
                           vertical - vertex_radius,
                           horizontal + vertex_radius,
                           vertical + vertex_radius,
                                    fill=color)
            self.canvas.create_text(horizontal, vertical + text_height, text=vertex, anchor="s",
                           fill="black", font=("Helvetica", 12))

        cvt_radius = 2
        cvt_extra_border = 40

        for vertex in graph.get_events():
            if self.apply_cutoff_to_vertices and vertex[1][self.layer] < self.cutoff_percentage:
                continue
            vertex = vertex[0]

            if vertex in self.current_location_map:
                vertical = self.current_location_map[vertex][1]
                horizontal = self.current_location_map[vertex][0]
            else:
                vertical = random.randint(forest_top + cvt_extra_border, forest_bottom - cvt_extra_border)
                horizontal = random.randint(forest_left + cvt_extra_border, forest_right - cvt_extra_border)
                self.current_location_map[vertex] = [horizontal, vertical]

            self.canvas.create_rectangle(horizontal - cvt_radius,
                                vertical - cvt_radius,
                                horizontal + cvt_radius,
                                vertical + cvt_radius)

        self.draw_edges()

    def draw_edges(self):
        graph = self.examples[self.example_pointer][1]
        if self.edge_objects is not None:
            for e in self.edge_objects:
                self.canvas.delete(e)

        self.edge_objects = []
        for edge in graph.get_all_edges():

            forward_gate = [float(e) for e in edge[3].split("/")]
            backward_gate = [float(e) for e in edge[4].split("/")]

            if forward_gate[self.layer] < self.cutoff_percentage and backward_gate[self.layer] < self.cutoff_percentage:
                continue
            elif forward_gate[self.layer] > self.cutoff_percentage and backward_gate[self.layer] > self.cutoff_percentage:
                arrows = tk.BOTH
            elif forward_gate[self.layer] > self.cutoff_percentage:
                arrows = tk.LAST
            else:
                arrows = tk.FIRST

            location_subject = self.current_location_map[edge[0]]
            location_object = self.current_location_map[edge[2]]

            line = self.canvas.create_line(location_subject[0], location_subject[1], location_object[0], location_object[1],
                                  arrow=arrows)

            middle_horizontal = int(location_subject[0] + location_object[0]) / 2
            middle_vertical = int(location_subject[1] + location_object[1]) / 2

            text = self.canvas.create_text(middle_horizontal, middle_vertical + 4, text=edge[1], anchor="s",
                                  fill="black", font=("Helvetica", 8))

            self.edge_objects.append(line)
            self.edge_objects.append(text)


analysis_tool = AnalysisTool(2)
analysis_tool.read_examples(examples)
analysis_tool.run()
from models.tensorflow_components.sentence.multihead_attention import MultiheadAttention
from models.tensorflow_components.sentence.word_padder import WordPadder


class DummyVertexFinalSentenceEmbedding:

    graph = None

    def __init__(self, graph, experiment_configuration):
        self.graph = graph
        self.variables = {}

    def run(self, mode):
        return self.graph.get_sentence_embeddings()

    def get_regularization(self):
        return 0

    def handle_variable_assignment(self, batch, mode):
        pass
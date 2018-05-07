from models.tensorflow_components.sentence.multihead_attention import MultiheadAttention
from models.tensorflow_components.sentence.word_padder import WordPadder


class WordDummyAttentionFinalSentenceEmbedding:

    graph = None

    def __init__(self, graph, experiment_configuration):
        self.graph = graph
        self.variables = {}

        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])
        attention_heads = 1
        self.attention_component = MultiheadAttention(gcn_dim, attention_heads=attention_heads, variable_prefix="attention2")
        self.word_padder = WordPadder()

    def run(self, mode):
        final_word_embeddings = self.graph.get_word_vertex_embeddings()
        padded_final_word_embeddings = self.word_padder.pad(final_word_embeddings)

        return self.attention_component.attend(padded_final_word_embeddings, mode)

    def get_regularization(self):
        return self.attention_component.get_regularization()

    def handle_variable_assignment(self, batch, mode):
        self.attention_component.handle_variable_assignment(batch, mode)
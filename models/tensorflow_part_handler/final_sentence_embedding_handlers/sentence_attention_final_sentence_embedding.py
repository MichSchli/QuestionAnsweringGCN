from models.tensorflow_components.sentence.multihead_attention import MultiheadAttention


class SentenceAttentionFinalSentenceEmbedding:

    sentence = None

    def __init__(self, sentence, experiment_configuration):
        self.sentence = sentence
        self.variables = {}

        lstm_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        attention_heads = int(experiment_configuration["lstm"]["attention_heads"])
        input_dim = lstm_dim * 2
        self.attention_component = MultiheadAttention(input_dim, attention_heads=attention_heads, variable_prefix="attention2")

    def run(self, mode):
        word_embeddings = self.sentence.get_word_embeddings()
        final_sentence_embedding = self.attention_component.attend(word_embeddings, mode)
        return final_sentence_embedding

    def get_regularization(self):
        return self.attention_component.get_regularization()

    def handle_variable_assignment(self, batch, mode):
        self.attention_component.handle_variable_assignment(batch, mode)
from models.dummy_model import DummyModel
from models.dummy_tensorflow_model import DummyTensorflowModel
from models.tensorflow_components.aux_components.sentence_to_entity_mapper import SentenceToEntityMapper
from models.tensorflow_components.gcn.gcn_factory import GcnFactory
from models.tensorflow_components.graph.graph_component import GraphComponent
from models.tensorflow_components.loss_functions.max_pred_sigmoid_loss import MaxPredSigmoidLoss
from models.tensorflow_components.loss_functions.sigmoid_loss import SigmoidLoss
from models.tensorflow_components.sentence.lstm import BiLstm
from models.tensorflow_components.sentence.multihead_attention import MultiheadAttention
from models.tensorflow_components.sentence.sentence_batch_component import SentenceBatchComponent
from models.tensorflow_components.sentence.word_padder import WordPadder
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent
from models.tensorflow_models.separate_lstm_vs_gcn import SeparateLstmVsGcn
from models.tensorflow_part_handler.final_sentence_embedding_handlers.dummy_vertex_final_sentence_embedding import \
    DummyVertexFinalSentenceEmbedding
from models.tensorflow_part_handler.final_sentence_embedding_handlers.sentence_attention_final_sentence_embedding import \
    SentenceAttentionFinalSentenceEmbedding
from models.tensorflow_part_handler.final_sentence_embedding_handlers.word_dummy_attention_final_sentence_embedding import \
    WordDummyAttentionFinalSentenceEmbedding


class ModelFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory
        self.gcn_factory = GcnFactory(index_factory)

    def get_separate_lstm_vs_gcn(self, experiment_configuration):
        word_index = self.index_factory.get("words", experiment_configuration)
        pos_index = self.index_factory.get("pos", experiment_configuration)

        model = SeparateLstmVsGcn()
        learning_rate = float(experiment_configuration["training"]["learning_rate"])
        gradient_clipping = float(experiment_configuration["training"]["gradient_clipping"])
        model.learning_rate = learning_rate
        model.gradient_clipping = gradient_clipping

        model.graph = GraphComponent()
        model.add_component(model.graph)

        model.sentence = SentenceBatchComponent(word_index,
                                                pos_index,
                                                word_dropout_rate=float(experiment_configuration["regularization"]["word_dropout"]))
        self.add_lstms(experiment_configuration, model)

        self.add_final_transform(experiment_configuration, model)

        model.loss = SigmoidLoss()
        model.add_component(model.loss)

        return model

    def get(self, experiment_configuration):
        model_type = experiment_configuration["architecture"]["model_type"]

        if model_type == "separate_lstm_vs_gcn":
            return self.get_separate_lstm_vs_gcn(experiment_configuration)

        word_index = self.index_factory.get("words", experiment_configuration)
        pos_index = self.index_factory.get("pos", experiment_configuration)

        model = DummyTensorflowModel()
        model.graph = GraphComponent()
        model.add_component(model.graph)
        model.add_component(model.graph.mention_dummy_assignment_view)
        model.add_component(model.graph.word_assignment_view)
        model.word_padder = WordPadder()
        model.add_component(model.word_padder)

        model.dummy_mlp = MultilayerPerceptronComponent([1, int(experiment_configuration["gcn"]["embedding_dimension"])], "dummy_mlp")
        model.add_component(model.dummy_mlp)

        model.word_node_init_mlp = MultilayerPerceptronComponent(
            [int(experiment_configuration["lstm"]["embedding_dimension"])*2,
             int(experiment_configuration["gcn"]["embedding_dimension"])],
            "word_node_init_mlp")
        model.add_component(model.word_node_init_mlp)

        model.sentence = SentenceBatchComponent(word_index,
                                                pos_index,
                                                word_dropout_rate=float(experiment_configuration["regularization"]["word_dropout"]),
                                                is_static=experiment_configuration["lstm"]["static_word_embeddings"] == "True")
        model.add_component(model.sentence)

        self.add_final_transform(experiment_configuration, model)

        model.sentence_to_entity_mapper = SentenceToEntityMapper(comparison_type="multiple")
        model.add_component(model.sentence_to_entity_mapper)

        model.loss = MaxPredSigmoidLoss()
        model.add_component(model.loss)

        learning_rate = float(experiment_configuration["training"]["learning_rate"])
        gradient_clipping = float(experiment_configuration["training"]["gradient_clipping"])

        model.learning_rate = learning_rate
        model.gradient_clipping = gradient_clipping

        model.gcn, model.sentence_batch_features = self.gcn_factory.get_gated_message_bias_gcn(model.graph, experiment_configuration)
        model.add_component(model.gcn)
        model.add_component(model.sentence_batch_features)

        self.add_lstms(experiment_configuration, model)

        if experiment_configuration["architecture"]["final_sentence_embedding"] == "sentence_attention":
            model.final_sentence_embedding = SentenceAttentionFinalSentenceEmbedding(model.sentence, experiment_configuration)
            model.add_component(model.final_sentence_embedding)
        elif experiment_configuration["architecture"]["final_sentence_embedding"] == "dummy_attention":
            model.final_sentence_embedding = WordDummyAttentionFinalSentenceEmbedding(model.graph, experiment_configuration)
            model.add_component(model.final_sentence_embedding)
            model.add_component(model.final_sentence_embedding.word_padder)
        elif experiment_configuration["architecture"]["final_sentence_embedding"] == "dummy_vertex":
            model.final_sentence_embedding = DummyVertexFinalSentenceEmbedding(model.graph, experiment_configuration)

        return model

    def add_lstms(self, experiment_configuration, model):
        word_dim = int(experiment_configuration["indexes"]["word_index_type"].split(":")[1])
        pos_dim = int(experiment_configuration["indexes"]["pos_index_type"].split(":")[1])
        lstm_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        lstm_layers = int(experiment_configuration["lstm"]["layers"])
        attention_heads = int(experiment_configuration["lstm"]["attention_heads"])
        attention_dropout = float(experiment_configuration["regularization"]["attention_dropout"])

        model.lstms = []
        for layer in range(lstm_layers):
            input_dim = word_dim + pos_dim + 1 if layer == 0 else lstm_dim * 2
            lstm = BiLstm(input_dim, lstm_dim*2, "bi_lstm_"+str(layer))
            model.lstms.append(lstm)
            model.add_component(lstm)

        input_dim = word_dim + pos_dim if lstm_layers == 0 else lstm_dim * 2

        model.gate_attention = MultiheadAttention(input_dim, attention_heads=attention_heads, variable_prefix="attention1", attention_dropout=attention_dropout)
        model.final_attention = MultiheadAttention(input_dim, attention_heads=attention_heads, variable_prefix="attention2", attention_dropout=attention_dropout)

        model.add_component(model.gate_attention)
        model.add_component(model.final_attention)

    def add_final_transform(self, experiment_configuration, model):

        if experiment_configuration["architecture"]["final_sentence_embedding"] == "sentence_attention":
            word_in_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        elif experiment_configuration["architecture"]["final_sentence_embedding"] == "dummy_attention":
            word_in_dim = int(int(experiment_configuration["gcn"]["embedding_dimension"])/2)
        elif experiment_configuration["architecture"]["final_sentence_embedding"] == "dummy_vertex":
            word_in_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        in_dim = gcn_dim * 3
        hidden_dims = [int(d) for d in experiment_configuration["other"]["final_hidden_dimensions"].split("|")]
        dropout_rate = float(experiment_configuration["regularization"]["final_dropout"])
        l2_rate = float(experiment_configuration["regularization"]["final_l2"])

        hidden_dims = [in_dim] + hidden_dims + [1]

        model.mlp = MultilayerPerceptronComponent(hidden_dims,
                                                  "final_mlp",
                                                  dropout_rate=dropout_rate,
                                                  l2_scale=l2_rate)
        model.add_component(model.mlp)
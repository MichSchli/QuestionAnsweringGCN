from models.dummy_model import DummyModel
from models.dummy_tensorflow_model import DummyTensorflowModel
from models.tensorflow_components.aux_components.sentence_to_entity_mapper import SentenceToEntityMapper
from models.tensorflow_components.gcn.gcn_factory import GcnFactory
from models.tensorflow_components.graph.graph_component import GraphComponent
from models.tensorflow_components.loss_functions.sigmoid_loss import SigmoidLoss
from models.tensorflow_components.sentence.lstm import BiLstm
from models.tensorflow_components.sentence.multihead_attention import MultiheadAttention
from models.tensorflow_components.sentence.sentence_batch_component import SentenceBatchComponent
from models.tensorflow_components.sentence.word_padder import WordPadder
from models.tensorflow_components.transformations.multilayer_perceptron import MultilayerPerceptronComponent


class ModelFactory:

    index_factory = None

    def __init__(self, index_factory):
        self.index_factory = index_factory
        self.gcn_factory = GcnFactory(index_factory)

    def get(self, experiment_configuration):
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

        model.sentence = SentenceBatchComponent(word_index, pos_index, word_dropout_rate=float(experiment_configuration["regularization"]["word_dropout"]))
        model.add_component(model.sentence)

        self.add_final_transform(experiment_configuration, model)

        model.sentence_to_entity_mapper = SentenceToEntityMapper()
        model.add_component(model.sentence_to_entity_mapper)

        model.loss = SigmoidLoss()
        model.add_component(model.loss)

        learning_rate = float(experiment_configuration["training"]["learning_rate"])
        gradient_clipping = float(experiment_configuration["training"]["gradient_clipping"])

        model.learning_rate = learning_rate
        model.gradient_clipping = gradient_clipping

        model.gcn_layers, model.sentence_batch_features = self.gcn_factory.get_dummy_gcn(model.graph, experiment_configuration)
        for gcn in model.gcn_layers:
            model.add_component(gcn)
        model.add_component(model.sentence_batch_features)

        self.add_lstms(experiment_configuration, model)

        return model

    def add_lstms(self, experiment_configuration, model):
        word_dim = int(experiment_configuration["indexes"]["word_index_type"].split(":")[1])
        pos_dim = int(experiment_configuration["indexes"]["pos_index_type"].split(":")[1])
        lstm_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        lstm_layers = int(experiment_configuration["lstm"]["layers"])
        attention_heads = int(experiment_configuration["lstm"]["attention_heads"])


        model.lstms = []
        for layer in range(lstm_layers):
            input_dim = word_dim + pos_dim if layer == 0 else lstm_dim * 2
            lstm = BiLstm(input_dim, lstm_dim*2, "bi_lstm_"+str(layer))
            model.lstms.append(lstm)
            model.add_component(lstm)

        input_dim = word_dim + pos_dim if lstm_layers == 0 else lstm_dim * 2

        model.gate_attention = MultiheadAttention(input_dim, attention_heads=attention_heads, variable_prefix="attention1")
        model.final_attention = MultiheadAttention(input_dim, attention_heads=attention_heads, variable_prefix="attention2")

        model.add_component(model.gate_attention)
        model.add_component(model.final_attention)

    def add_final_transform(self, experiment_configuration, model):
        lstm_dim = int(experiment_configuration["lstm"]["embedding_dimension"])
        gcn_dim = int(experiment_configuration["gcn"]["embedding_dimension"])

        in_dim = gcn_dim + lstm_dim
        hidden_dims = [int(d) for d in experiment_configuration["other"]["final_hidden_dimensions"].split("|")]
        dropout_rate = float(experiment_configuration["regularization"]["final_dropout"])
        l2_rate = float(experiment_configuration["regularization"]["final_l2"])

        hidden_dims = [in_dim] + hidden_dims + [1]

        model.mlp = MultilayerPerceptronComponent(hidden_dims,
                                                  "mlp",
                                                  dropout_rate=dropout_rate,
                                                  l2_scale=l2_rate)
        model.add_component(model.mlp)
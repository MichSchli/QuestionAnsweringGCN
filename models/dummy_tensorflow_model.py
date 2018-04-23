from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class DummyTensorflowModel(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        word_embeddings = self.sentence.get_embedding()
        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)

        dummy_counts = self.graph.get_dummy_counts()
        dummy_embeddings = self.dummy_mlp.transform(tf.expand_dims(dummy_counts, -1), mode)

        self.graph.initialize_zero_embeddings(tf.shape(dummy_embeddings)[-1])
        self.graph.set_dummy_embeddings(dummy_embeddings)
        self.graph.set_word_embeddings(word_embeddings)

        gate_sentence_embedding = self.gate_attention.attend(word_embeddings, mode)
        self.sentence_batch_features.set_batch_features(gate_sentence_embedding)

        gcn_cell_state = None
        for gcn in self.gcn_layers:
            gcn_cell_state = gcn.propagate(mode, gcn_cell_state)

        entity_embeddings = self.graph.get_target_vertex_embeddings()
        final_sentence_embedding = self.graph.get_sentence_embeddings()

        final_word_embeddings = self.graph.get_word_vertex_embeddings()
        padded_final_word_embeddings = self.word_padder.pad(final_word_embeddings)
        final_sentence_embedding = self.final_attention.attend(padded_final_word_embeddings, mode)

        entity_embeddings = self.sentence_to_entity_mapper.map(final_sentence_embedding, entity_embeddings)
        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return tf.squeeze(entity_embeddings)

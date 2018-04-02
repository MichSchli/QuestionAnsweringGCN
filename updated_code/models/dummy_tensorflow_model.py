from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class DummyTensorflowModel(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        word_embeddings = self.sentence.get_embedding()
        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)

        gate_sentence_embedding = self.gate_attention.attend(word_embeddings, mode)
        final_sentence_embedding = self.final_attention.attend(word_embeddings, mode)

        #self.graph.initialize_zero_embeddings(dimension=1)
        self.graph.initialize_dummy_counts()
        self.sentence_batch_features.set_batch_features(gate_sentence_embedding)

        gcn_cell_state = None
        for gcn in self.gcn_layers:
            gcn_cell_state = gcn.propagate(mode, gcn_cell_state)

        entity_embeddings = self.graph.get_target_vertex_embeddings()

        entity_embeddings = self.sentence_to_entity_mapper.map(final_sentence_embedding, entity_embeddings)
        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return tf.squeeze(entity_embeddings)

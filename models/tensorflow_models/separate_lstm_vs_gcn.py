from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class SeparateLstmVsGcn(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        word_embeddings = self.sentence.get_embedding()
        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)

        self.graph.initialize_zero_embeddings()

        gcn_cell_state = self.initial_cell_updater.initialize(self.graph.get_full_vertex_embeddings())
        for gcn in self.gcn_layers:
            gcn_cell_state = gcn.propagate(mode, gcn_cell_state)

        entity_embeddings = self.graph.get_target_vertex_embeddings()
        final_sentence_embedding = self.final_attention.attend(word_embeddings, mode)

        entity_embeddings = self.sentence_to_entity_mapper.map(final_sentence_embedding, entity_embeddings)
        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return tf.squeeze(entity_embeddings)

from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class DummyTensorflowModel(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        word_embeddings = self.sentence.get_embedding()
        sentence_embedding = tf.reduce_mean(word_embeddings, axis=1)

        self.graph.initialize_zero_embeddings(dimension=1)
        self.sentence_batch_features.set_batch_features(sentence_embedding)
        self.gcn.propagate(mode)

        entity_embeddings = self.graph.get_target_vertex_embeddings()

        entity_embeddings = self.sentence_to_entity_mapper.map(sentence_embedding, entity_embeddings)
        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return tf.squeeze(entity_embeddings)

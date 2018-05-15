from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class DummyTensorflowModel(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        word_embeddings = self.sentence.get_embedding()
        for lstm in self.lstms:
            word_embeddings = lstm.transform_sequences(word_embeddings)
        self.sentence.current_word_embeddings = word_embeddings

        dummy_counts = self.graph.get_dummy_counts()
        dummy_embeddings = self.dummy_mlp.transform(tf.expand_dims(dummy_counts, -1), mode)

        self.graph.initialize_zero_embeddings(tf.shape(dummy_embeddings)[-1])
        self.graph.set_dummy_embeddings(dummy_embeddings)

        word_embedding_shape = tf.shape(self.sentence.current_word_embeddings)

        word_node_init_embeddings = tf.reshape(self.sentence.current_word_embeddings, [-1, word_embedding_shape[-1]])
        word_node_init_embeddings = self.word_node_init_mlp.transform(word_node_init_embeddings, mode)

        #self.graph.set_word_embeddings(word_node_init_embeddings, reshape=False)

        gate_sentence_embedding = self.gate_attention.attend(word_embeddings, mode)
        self.sentence_batch_features.set_batch_features(gate_sentence_embedding)

        self.gcn.run(mode)

        entity_embeddings = self.graph.get_target_vertex_embeddings()

        #entity_embeddings = tf.Print(entity_embeddings, [entity_embeddings], message="entities: ", summarize=200)

        final_sentence_embedding = self.final_sentence_embedding.run(mode)

        entity_embeddings = self.sentence_to_entity_mapper.map(final_sentence_embedding, entity_embeddings, mode)
        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return tf.squeeze(entity_embeddings)

import tensorflow as tf


class SentenceBatchComponent:

    word_embeddings = None
    pos_embeddings = None

    variables = None
    variable_assignments = None
    index = None

    def __init__(self, word_index, pos_index, word_dropout_rate=0.0, is_static=False):
        self.variables = {}
        self.word_index = word_index
        self.pos_index = pos_index

        self.word_dropout_rate = word_dropout_rate
        self.is_static = is_static

        self.prepare_tensorflow_variables()

    def prepare_tensorflow_variables(self):
        self.variables["word_indices"] = tf.placeholder(tf.int32, [None, None])
        self.variables["pos_indices"] = tf.placeholder(tf.int32, [None, None])
        self.variables["word_mask"] = tf.placeholder(tf.float32, [None, None])

        initializer = self.word_index.get_all_vectors()
        self.word_embeddings = tf.Variable(initializer, trainable=not self.is_static)

        initializer = self.pos_index.get_all_vectors()
        self.pos_embeddings = tf.Variable(initializer)

    def get_embedding(self, mode="train"):
        word_embedding = tf.nn.embedding_lookup(self.word_embeddings, self.variables["word_indices"])
        pos_embedding = tf.nn.embedding_lookup(self.pos_embeddings, self.variables["pos_indices"])

        embedding = tf.concat([word_embedding, pos_embedding], axis=-1)

        if mode == "train" and self.word_dropout_rate > 0.0:
            embedding_shape = tf.shape(embedding)
            noise_shape = [embedding_shape[0], embedding_shape[1], 1]
            embedding = tf.nn.dropout(embedding, 1 - self.word_dropout_rate, noise_shape=noise_shape)
        return embedding * tf.expand_dims(self.variables["word_mask"], -1)

    def get_variable(self, name):
        return self.variables[name]

    def handle_variable_assignment(self, batch, mode):
        self.variable_assignments = {}

        self.variable_assignments["word_indices"] = batch.get_padded_word_indices()
        self.variable_assignments["pos_indices"] = batch.get_padded_pos_indices()
        self.variable_assignments["word_mask"] = batch.get_word_padding_mask()
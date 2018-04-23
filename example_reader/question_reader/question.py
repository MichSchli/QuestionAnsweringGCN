class Question:

    words = None
    word_indexes = None
    pos = None
    pos_indexes = None

    def count_words(self):
        return self.word_indexes.shape[0]

    def get_word_indexes(self):
        return self.word_indexes

    def get_pos_indexes(self):
        return self.pos_indexes
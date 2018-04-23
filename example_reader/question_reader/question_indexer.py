import numpy as np


class QuestionIndexer:

    inner = None
    word_index = None
    pos_index = None

    def __init__(self, inner, word_index, pos_index):
        self.inner = inner
        self.word_index = word_index
        self.pos_index = pos_index

    def build(self, array_question):
        question = self.inner.build(array_question)
        question.word_indexes = np.array([self.word_index.index(word) for word in question.words])
        question.pos_indexes = np.array([self.pos_index.index(pos) for pos in question.pos])
        return question
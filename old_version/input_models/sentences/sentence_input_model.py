class SentenceInputModel:

    word_index_matrix = None
    sentence_lengths = None
    centroid_map = None
    n_centroids = None

    def get_max_words_in_batch(self):
        return self.sentence_lengths.max()
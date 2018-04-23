class Mention:

    word_indexes = None
    entity_label = None
    entity_index = None
    score = None

    def __str__(self):
        return "Mention: \""+str(self.entity_label)+"\" / "+str(self.entity_index)+" ("+str(self.word_indexes[0])+"-"+str(self.word_indexes[-1])+")"
class GoldAnswer:

    entity_name_or_label = None
    entity_indexes = None

    def __str__(self):
        return "Gold: "+self.entity_name_or_label + " / " + (", ".join([str(e) for e in self.entity_indexes]) if self.entity_indexes is not None else "None")
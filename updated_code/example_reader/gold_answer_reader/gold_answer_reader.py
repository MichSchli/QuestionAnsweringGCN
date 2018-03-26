from example_reader.gold_answer_reader.gold_answer import GoldAnswer


class GoldAnswerReader:

    def __init__(self):
        pass

    def build(self, array_gold):
        gold_answers = []
        for gold_answer_line in array_gold:
            gold_answer = GoldAnswer()
            gold_answer.entity_name_or_label = gold_answer_line[1]
            gold_answers.append(gold_answer)
        return gold_answers
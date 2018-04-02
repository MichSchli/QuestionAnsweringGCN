from example_reader.question_reader.question import Question


class QuestionReader:

    dataset_map = None
    dataset = None

    def __init__(self):
        pass

    def build(self, array_question):
        question = Question()
        question.words = [array_question[j][1] for j in range(len(array_question))]
        question.pos = [array_question[j][3] for j in range(len(array_question))]
        question.dep_types = [array_question[j][4] for j in range(len(array_question))]
        question.dep_heads = [int(array_question[j][5]) for j in range(len(array_question))]

        return question
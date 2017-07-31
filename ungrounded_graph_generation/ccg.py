class DerivationLeaf:

    word = None
    lemma = None
    pos = None
    ner = None
    ccg_tag = None

    def pretty_print(self, indent=0):
        string = ''
        for i in range(indent):
            string += '  '

        string += self.word + ' | ' + self.ccg_tag
        print(string)


class DerivationElement:

    ccg_type = None
    ccg_tag = None
    left_child = None
    right_child = None


    def pretty_print(self, indent=0):
        string = ''
        for i in range(indent):
            string += '  '

        string += self.ccg_type + ' | ' + self.ccg_tag
        print(string)

        if self.left_child is not None:
            self.left_child.pretty_print(indent=indent+1)

        if self.right_child is not None:
            self.right_child.pretty_print(indent=indent+1)



class ParseElement:

    text = None
    left_child = None
    right_child = None

    def __init__(self, text):
        self.text = text

    def to_ccg_derivation(self):
        text_parts = self.text.split(' ')
        if self.left_child is None:
            element = DerivationLeaf()
            element.word = text_parts[2]
            element.lemma = text_parts[3]
            element.pos = text_parts[4]
            element.ner = text_parts[5]
            element.ccg_tag = text_parts[1]

            return element
        else:
            element = DerivationElement()
            element.ccg_tag = text_parts[1]
            element.ccg_type = text_parts[2]
            element.left_child = self.left_child.to_ccg_derivation()

            if self.right_child is not None:
                element.right_child = self.right_child.to_ccg_derivation()

            return element

    def pretty_print(self, indent=0):
        string = ''
        for i in range(indent):
            string += '  '

        string += self.text
        print(string)

        if self.left_child is not None:
            self.left_child.pretty_print(indent=indent+1)

        if self.right_child is not None:
            self.right_child.pretty_print(indent=indent+1)

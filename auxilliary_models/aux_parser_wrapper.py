class AuxParserWrapper:

    filename = None
    parser = None

    def __init__(self, parser, filename):
        self.filename = filename
        self.parser = parser

    def get_iterator(self):
        return self.parser.parse_file(self.filename)
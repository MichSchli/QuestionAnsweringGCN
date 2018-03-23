import math


class SettingsReader:

    def read(self, settings_file):
        f = open(settings_file, 'r')
        dictionary, _ = self.internal_parse(0, list(f))
        return dictionary

    def internal_parse(self, start_index, lines, indent=0):
        index = start_index
        dictionary = {}
        while index < len(lines):
            line = lines[index]
            if line.strip():
                indent_level = self.__count_indents__(line)

                if indent_level < indent:
                    return dictionary, index

                line = line.strip()

                if line.startswith('['):
                    sub_dictionary, sub_end_index = self.internal_parse(index + 1, lines, indent=indent + 1)
                    dictionary[line[1:-1]] = sub_dictionary
                    index = sub_end_index
                else:
                    parts = [p.strip() for p in line.split('=')]
                    dictionary[parts[0]] = parts[1]
                    index += 1
            else:
                index += 1

        return dictionary, index

    def __count_indents__(self, line):
        space = 0
        tab = 0
        for c in line:
            if c == " ":
                space += 1
            elif c == "t":
                tab += 1
            else:
                return tab + math.floor(space/3)
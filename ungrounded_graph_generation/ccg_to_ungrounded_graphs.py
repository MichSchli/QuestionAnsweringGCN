import argparse
import json
import sys

from ccg import ParseElement


def line_to_graph(line):
    pass


def text_to_parse_tree(line):
    stack = []
    pointer = 0

    while pointer < len(line):
        if line[pointer] == '(':
            opposite = line.index('>', pointer)
            parse_element = ParseElement(line[pointer + 2:opposite])
            pointer = opposite

            if not stack:
                stack.append(parse_element)
            elif stack[-1].left_child is None:
                stack[-1].left_child = parse_element
                stack.append(parse_element)
            else:
                stack[-1].right_child = parse_element
                stack.append(parse_element)
        elif line[pointer] == ')':
            most_recent = stack.pop()

        pointer += 1

    return most_recent


def add_indexing(ccg_derivation):
    return ccg_derivation


def parse_from_console():
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("ID"):
            print(line)
        else:
            ccg_derivation = text_to_parse_tree(line)#.to_ccg_derivation()
            ccg_derivation = add_indexing(ccg_derivation)
            ccg_derivation.pretty_print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Formats and yields json data to stdout for easyccg parsing.')

    args = parser.parse_args()

    parse_from_console()


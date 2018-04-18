#!/bin/bash

INPUT=$1
OUTPUT=$2

bash scripts/recase.sh $INPUT

python3 helpers/alignment_group_adder.py --file $INPUT.uppercased
python3 helpers/add_pos_and_dep.py --input_file $INPUT.uppercased > $OUTPUT

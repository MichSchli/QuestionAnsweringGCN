#!/bin/bash

(cd auxilliary_models/siva-parser/ && cat data/spades/train.json.blank.cleaned | bash run.sh)
echo "END"

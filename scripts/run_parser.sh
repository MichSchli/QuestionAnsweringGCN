#!/bin/bash

python3 text_parsing/json_to_easyccg.py | java -jar text_parsing/easyccg.jar --model text_parsing/models/easyccg_model --nbest 10 -i POSandNERtagged -o extended

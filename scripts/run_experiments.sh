#!/usr/bin/env bash

CONFIGURATIONS=(configurations/model/bow+dumb+transform.cfg configurations/model/lstm+dumb+transform.cfg)
DATASETS=(configurations/dataset/toy-125.cfg)
PYTHON_INTERPRETER="python3"

for model_configuration in ${CONFIGURATIONS[@]}; do
    for dataset_configuration in ${DATASETS[@]}; do
        echo "Running experiment for "$model_configuration" using "$dataset_configuration
        $PYTHON_INTERPRETER parameter_search.py --algorithm $model_configuration --dataset $dataset_configuration
    done
done
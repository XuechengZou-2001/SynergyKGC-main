#!/usr/bin/env bash

set -x

TASK="FB15k237"

if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "./data/${TASK}/train.txt" \
--valid-path "./data/${TASK}/valid.txt" \
--test-path "./data/${TASK}/test.txt"


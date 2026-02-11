#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"

model_path="Please replace with your actual folder name."
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

test_path="${DATA_DIR}/test.txt.json"

neighbor_weight=0.05
rerank_n_hop=2

python3 -u evaluate.py \
--task "${TASK}" \
--is-test \
--eval-model-path "${model_path}" \
--pretrained-model "Please replace with your actual folder name." \
--gnn_start_epoch 5 \
--density-threshold 1 \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
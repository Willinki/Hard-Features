#!/bin/bash

betas=(0.5 1.0 2.0 5.0 10.0)
weight_decays=(0.00001 0.0001 0.001 0.01 0.1)

for beta in "${betas[@]}"; do
    for weight_decay in "${weight_decays[@]}"; do
        echo "Running with model.beta=$beta and model.weight_decay=$weight_decay"
        python3 scripts/train_tanh_mlp.py beta=$beta weight_decay=$weight_decay
    done
done
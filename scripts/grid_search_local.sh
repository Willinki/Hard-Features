#!/bin/bash

betas=(1000)
binary=(true)

for beta in "${betas[@]}"; do
    for binary in "${binary[@]}"; do
        echo "Running with model.beta=$beta and model.binary=$binary"
        python3 scripts/train_tanh_mlp.py model.beta=$beta model.binary=$binary trainer.max_epochs=200 model.lr=5e-5
    done
done
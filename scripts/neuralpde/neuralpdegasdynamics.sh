#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py model=neuralPDE datamodule=gas_dynamics name=run_1 model.net.input_dim=4 trainer.max_epochs=5 +trainer.val_check_interval=0.2
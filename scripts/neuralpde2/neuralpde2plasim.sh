#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py model=neuralPDE_2order datamodule=plasim name=run_1 model.net.input_dim=4 trainer.max_epochs=5 +trainer.val_check_interval=0.2 trainer.resume_from_checkpoint=/home/ls6/dulny/neuralPDE-2022/logs2/experiments/runs/plasim/neuralPDE_2order/2022-05-23_10-30-34/checkpoints/epoch_001.ckpt
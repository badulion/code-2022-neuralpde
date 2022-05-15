#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python train.py model=persistence name=persistence train=false

#python train.py trainer.max_epochs=50 model=pdernn name=pdernn_history model.use_last_only=false trainer.gpus=1

#python train.py trainer.max_epochs=50 model=pdernn name=pdernn_one model.use_last_only=true trainer.gpus=1

#python train.py trainer.max_epochs=20 model=resnet name=resnet trainer.gpus=1

#python train.py trainer.max_epochs=20 model=convLSTM name=convLSTM trainer.gpus=1

python train.py trainer.max_epochs=20 model=distana name=distana trainer.gpus=1

#python train.py trainer.max_epochs=20 model=pdenet name=pdenet trainer.gpus=1
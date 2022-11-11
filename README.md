This repository contains the code for our paper "NeuralPDE: Modelling Dynamical Systems from Data".


# Running the experiments

To run the experiments run 

> python train.py model=MODEL datamodule=DATASET

where `MODEL`and `DATASET` is one of

| MODEL  | DATASET  |
|---|---|
| cnn  | burgers  |
| convLSTM | advection_diffusion |
| distana  | gas_dynamics  |
| hiddenstate | advection_diffusion |
| pdenet |plasim |
| persistence  | wave  |
| resnet | weatherbench |
| neuralPDE  |
| neuralPDE_2order  |



# Data

The data for the `Plasim`, `Weatherbench` and `Oceanwave` experiments needs to be downloaded separately

The data for the `Advection-Diffusion`, `Gas Dynamics`, `Burgers` and `Wave` equations can be generated by running:

> python generate.py equation=EQUATION


# Cite

Please cite our paper if you use this code in your own work:

    @InProceedings{dulny2022neuralPDE,
        author="Dulny, Andrzej and Hotho, Andreas and Krause, Anna",
        title="NeuralPDE: Modelling Dynamical Systems from Data",
        booktitle="KI 2022: Advances in Artificial Intelligence",
        year="2022",
        publisher="Springer International Publishing",
        address="Cham",
        pages="75--89"
    }
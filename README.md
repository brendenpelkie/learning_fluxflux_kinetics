# Learning Flux-Flux Kinetics


This is the accompanying code and data for the paper, "Machine learning the quantum flux-flux correlation function for catalytic surface reactions" (https://arxiv.org/abs/2206.01845). The goal of this work is to predict fully quantum one dimensional Minimum Energy Path (MEP) reaction rate constants for surface catalyzed reactions using machine learning. This repository contains tools to reproduce our flux-flux correlation function and rate constant calculations and machine learning results. 


## Repository Description

This repository is structured as follows. 

```
├─ surfreact/ : Module code containing flux-flux correlation function calculation functions and helper code for machine learning
  ├─ reactions.py: module for fitting reaction barriers and managing some parts of Cff calculation
  ├─ numpycff.py: module for calculating Cff(t) using numpy matrix operations
  ├─ flintcff.py: module for calculating Cff(t) using flint arbitrary precision arithmetic matrix operations. Required for some reactions.
  ├─ submitManagers.py: module for managing parallel submission of Cff(t) jobs on a slurm HPC system
  ├─ representations.py: module with utility functions for computing atomic representations for ML
  ├─ mlutils.py: utility functions for machine learning tasks
  ├─ utils.py: general utilities and physical constants definitions
├─ data/ : Contains reaction data for Cff calculations and Cff data for machine learning
  ├─ MLdataset/: contains Cff(t) data for machine learning tasks. Intended to be loaded with utility funtion in mlutils.py
  ├─ cff_dataset/: dataset containing barrier and DVR parameters for use in Cff(t) calculations
  ├─ geometries/: Full geometries of reactants, products, and transition states
├─ environment.yml files: Conda environment files. One for Cff(t) calculations, and one for machine learning.
├─ machine_learning: contains notebooks for training and evaluating ML models
  ├─ cfflearn: Models that learn the full Cff(t) function
    ├─ reaction_split: models with reaction-wise train/test split as described in paper
    ├─ temperature_split: models with temperature-wise train/test split as described in paper
  ├─ kQlearn: Models that learn the rate constant kQ directly
    ├─ subdirectories: see cfflearn item above
├─ notebooks: Jupyter notebooks used in processing
  ├─ cff_integration.ipynb: workflow for combining parallelized Cff(t) calculation results and integrating to get a rate constant
```

## Setup and Requirements

All code and data needed to reproduce our results is contained in this repository. To set it up on your machine, first clone it. Then, create the conda environments by running `conda create -f $FILENAME.yml`. With these environments activated, install our module into the environment by running `pip install .` from the root of this repository. 

Major Dependencies:

- pandas
- numpy
- pyflint 
- atomic simulation environment
- molml
- scikit-learn
- scipy
- matplotlib
- seaborn
- mpmath


** pyflint is used for arbitrary precision calculations, which are required by some reactions at low temperatures to avoid numerical divergence issues. pyfint requires that flint be installed, which in turn requires GMP and MPFR. Install details at https://www.flintlib.org/. 

## Reproducing our Results

### Reproducing Cff(t) calculations
Cff(t) calculations can be run using the `runcffcalculation.py` python script. After following the setup instructions above:

1. Select a reaction to run a Cff(t) calculation for, from the list of reactions in Table 1 of the accompanying publication. From the entry for that reaction in the cff_dataset, select a temperature from that reaction's list of assigned temperatures. Or, pick your own temperature. The minimum temperature that the DVR grid parameters for each reaction have been verified to is noted in this dataset. Running at a temperature below this temperature will return invalid results.

2. Open the `runcffcalculation.py` file in a text editor and update the reaction number, temperature, and filepath values (described in the file) to your selected values. You may also want to change the maximum time the calculation is run to, or run the calculation on an offset time grid. To get more accurate integrated rate constants, calculations for the published Cff(t) time series were run on a finer time grid for the initial time points. This is done by setting the timeOffset parameter in the script to `True`.

3. Run the calculation script in the `cff_calculation` conda environment.

4. Post-process the results using the `IntegrateCFF.ipynb` notebook.

A note on compute resource usage: Cff(t) calculations fall into two categories: those that require arbitary precision math and those that don't. At low temperatures, some reactions exhibit numerical stability issues that lead to Cff(t) divergence. This requires the use of high-precision matrix manipulations to solve, and comes at the expense of time and memory. Numpy-based calculations at double precision should run in minutes to hours, depending on the DVR grid size. Flint-based calculations may take up to several weeks CPU time to run, and may require up to 175GB of memory. These calculations were run by breaking on Cff(t) time series calculation into multiple separate time chunks, and were run on the UW's Hyak HPC system. 

### Reproducing our machine learning results

All machine learning related tasks including feature generation, model training, and evaluation were done in jupyter notebooks. There is one notebook for each case discussed in the paper located in its respective directory in `machine_learning`. To reproduce a machine learning workflow:

1. Open the appropriate notebook in the cfflearning environment.
2. Excecute the code in the notebook. 

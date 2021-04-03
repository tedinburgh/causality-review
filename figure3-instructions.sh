## CODECHECK notes
##
## These are the instructions for regenerating figure 3 of our paper.
## To regenerate all the data woudl take around 60 hourss, so we have
## selected on figure (3) to reproduce as this is the quickest.
##
## Requirements
##
## Anaconda with Python (at least 3.8.3).




## Steps

## Clone the repository

git clone https://github.com/tedinburgh/causality-review
cd causality-review

## To run our code, you will need python 3.8.3 at least together
## with extra packages; these are installed into a virtual environment:

conda create --name causality_test --file requirements.txt
conda activate causality_test

## Before running the simulations, we first remove the stored
## data files and figures.

rm simulation-data/*
rm figures/*

## The following job takes about 3 hours on a modern linux machine.
time python causality-review-code/model_simulations.py --sim lp

## After the simulation has run, we now copy the simulation results into the
## required folder.
mv causality-review-code/*.csv simulation-data
cd causality-review-code/
python misc.py

## The python code currently errors at line 519, or thereabouts, but figure
## 3 will have been generated in the figures/ folder.

## At the end, to close the virtual environment, you can simply do:

conda deactivate



######################################################################

## DAMTP specific instructions
## see also: https://www.maths.cam.ac.uk/computing/software/using-python
## sje30@macro:~ $ module load miniconda3
## sje30@macro:~ $ source activate anaconda-2020.11

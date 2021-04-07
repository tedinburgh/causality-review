## CODECHECK notes
## Paper: 
## Edinburgh T, Eglen SJ, Ercole A (2021)
## Causality indices for bivariate time series data: a comparative review
## of performance. arXiv [statME]
## Available at: http://arxiv.org/abs/2104.00718.
##
## ORCids:
## https://orcid.org/0000-0002-3599-7133
## https://orcid.org/0000-0001-8607-8025
## https://orcid.org/0000-0001-8350-8093
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
## data files (Linear process) and figures.

rm simulation-data/lp_*.csv
cp figures/methods_paper.eps .
rm figures/*
cp methods_paper.eps figures/

## The following job takes about 3 hours on a modern linux machine.
time python causality-review-code/model_simulations.py --sim lp

## The following job can take up to about 15 minutes on a modern linux machine.
time python causality-review-code/misc_ci.py

## At the end, to close the virtual environment, you can simply do:

conda deactivate


######################################################################

## DAMTP specific instructions
## see also: https://www.maths.cam.ac.uk/computing/software/using-python
## sje30@macro:~ $ module load miniconda3
## sje30@macro:~ $ source activate anaconda-2020.11

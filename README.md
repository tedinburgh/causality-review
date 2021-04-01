# causality-review
---
Code (python), figures and data for a review of the performance of causality indices for bivariate time series data. This review follows previous work by Lungarella et al. (2007).

Methods included in this review are:
* Extended Granger causality (Chen et al. 2004)
* Nonlinear Granger causality (Ancona et al. 2004)
* Predictability improvement (Feldmann and Bhattacharya 2004)
* Transfer entropy (histogram partition and Kraskov-Stögbauer-Grassberger estimate) (Schreiber 2000, Kraskov et al. 2004)
* Effective transfer entropy (histogram partition) (Marschinski and Kantz 2002)
* Coarse-grained transinformation rate (Palus et al. 2001)
* Similarity indices (Arnhold et al. 1999, Bhattacharya et al. 2003)
* Convergent cross mapping (Sugihara et al. 2012)

Simulated model systems included in this review are:
* Linear (Gaussian) process (Lungarella et al. 2007)
* Ulam lattice (e.g. Schreiber 2000, Lungarella et al. 2007)
* Hénon unidirectional maps and Hénon bidirectional maps (Hénon 1976)

## Usage

To clone and run this code, you'll need [Git](https://git-scm.com) and [conda](https://docs.conda.io) (or equivalent for package, dependency and environment management) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/tedinburgh/causality-review

# Go into the repository
$ cd causality-review

# Install dependencies
$ conda create --name causality_test --file requirements.txt
$ conda activate causality_test
```

To generate all figures from the .csv data files in this repository, run: 

```bash
# Run script for all figures
$ python causality-review-code/misc_ci.py
```

To generate .csv data files, run e.g.:

```bash
# Linear process simulations, all methods
$ python causality-review-code/model_simulations.py --sim lp

# Ulam lattice simulations, nonlinear Granger causality
$ python causality-review-code/model_simulations.py --sim ul --ind nlgc
```

## References

M. Lungarella, K. Ishiguro, Y. Kuniyoshi, and N. Otsu, “Methods for quantifying the causal structure of bivariate time series,” Int. J. Bifurcat. Chaos 17, 903–921 (2007)

Y. Chen, G. Rangarajan, J. Feng,  and M. Ding, “Analyzing multiple nonlinear time series with extended Granger causality,” Phys. Lett. A 324, 26–35 (2004)

N. Ancona, D. Marinazzo, and S. Stramaglia, “Radial basis function approach to nonlinear Granger causality of time series,” Phys. Rev. E Stat. Nonlin. Soft Matter Phys. 70, 056221 (2004)

U. Feldmann and J. Bhattacharya, “Predictability improvement as an asymmetrical measure of interdependence in bivariate time series,” Int. J. Bifurcat. Chaos 14, 505–514 (2004)

T. Schreiber, “Measuring information transfer,” Phys. Rev. Lett.85, 461–464 (2000)

A. Kraskov, H. Stögbauer, and P. Grassberger, “Estimating mutual information,” (2004)

R. Marschinski and H. Kantz, “Analysing the information flow between financial time series,” The European Physical Journal B - Condensed Matter and Complex Systems 30, 275–281 (2002)

M. Paluš, V. Komárek, Z. Hrncír, and K. Sterbová, “Synchronization as adjustment of information rates: detection from bivariate time series,” Phys. Rev. E Stat. Nonlin. Soft Matter Phys. 63, 046211 (2001)

J. Arnhold, P. Grassberger, K. Lehnertz, and C. E. Elger, “A robust method for detecting interdependences: application to intracranially recorded EEG,” Physica D 134, 419–430 (1999)

J. Bhattacharya, E. Pereda, and H. Petsche, “Effective detection of coupling in short and noisy bivariate data,” IEEE Trans. Syst. Man Cybern. B Cybern. 33, 85–95 (2003)

G. Sugihara,  R. May,  H. Ye,  C.-H. Hsieh,  E. Deyle,  M. Fogarty, and S. Munch, “Detecting causality in complex ecosystems,” Science 338, 496–500 (2012)

M. Hénon, “A two-dimensional mapping with a strange attractor,” Commun. Math. Phys.50, 69–77 (1976)

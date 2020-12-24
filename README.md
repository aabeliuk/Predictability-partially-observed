# Predictability-partially-observed
Repository for paper "Predictability limit of partially observed systems" by Abeliuk, A., Huang, Z., Ferrara, E., & Lerman, K, Sci Rep, 2020 (https://www.nature.com/articles/s41598-020-77091-1)


## Repo Structure

+ **`Epidemics.ipynb`** contains the source code of the Epidemics results.

+ **`Data`** contains the data for Epidemics results.
  + `Epidemics` has the weekly state-level data for all diseases originally compiled by the USA National Notifable Diseases Surveillance System. For the covariance experiment, we used infuenza data from 2010 to 2015 obtained for the US Outpatient
Infuenza-like Illness Surveillance Network (ILINet) that overlaps with Google Flu Trends Data

+ **`Results`** Folder where the predictabilty measures computed are saved.

+ **`Figures`** Folder where replication of the paper figures are saved.

## Dependencies

1. Python 3
2. [PyInform](https://github.com/elife-asu/pyinform) (A Python Wrapper for the Inform Information Analysis Library) to compute mutual information

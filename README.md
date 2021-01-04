# Predictability-partially-observed
Repository for paper "Predictability limit of partially observed systems" by Abeliuk, A., Huang, Z., Ferrara, E., & Lerman, K, Sci Rep, 2020 (https://www.nature.com/articles/s41598-020-77091-1)


## Repo Structure

+ **`Epidemics.ipynb`** contains the source code of the Epidemics results.

+ **`Twitter-user.ipynb`** contains the source code of the Twitter user-level analysis results.

+ **`Twitter-hashtag.ipynb`** contains the source code of the Twitter hashtag-level analysis results.

+ **`Data`** contains the data for Epidemics results.
  + `Epidemics` has the weekly state-level data for all diseases originally compiled by the USA National Notifable Diseases Surveillance System. For the covariance experiment, we used infuenza data from 2010 to 2015 obtained for the US Outpatient
Infuenza-like Illness Surveillance Network (ILINet) that overlaps with Google Flu Trends Data
  + `Twitter` the social media data used in this study collected from Twitter in 2014. Starting with a set of
100 users who were active discussing ballot initiatives during the 2012 California election, we expanded this set
by retrieving the accounts of the users they followed, for a total of 5599 seed users. We collected all posts made by
the seed users and their friends (i.e., users they followed on Twitter) over the period of June–November 2014, a
total of over 600 thousand users. 

+ **`Results`** Folder where the predictabilty measures computed are saved.

+ **`Figures`** Folder where replication of the paper figures are saved.

## Dependencies

1. Python 3

2. [PyInform](https://github.com/elife-asu/pyinform) (A Python Wrapper for the Inform Information Analysis Library) to compute mutual information

# **************************************************************************************
# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquePy/LiquePy
# **************************************************************************************

This open-source Python library is an attempt to facilitate research on soil liquefaction and lateral spreads by providing researchers/engineers with premade/verified Python codes.

If you are willing to contribute write an email to massoud.hosseinali@utah.edu

So far the following methods have been added:
  - lateralspreads/spt_based.py:
    - Multi Linear Regression (MLR) (Youd, Hansen, & Bartlett 2002)
    - Multi Linear Regression (MLR) (Bardet et al. 2002)
    - Genetic programming (Javadi et al. 2006)
    - Evolutionary-based approach (Rezania et al. 2011)
    - Artificial Neural Network & Genetic Algorithm (Baziar & Azizkani 2013) *Read below
    - Multivariate Adaptive Regression Splines (MARS) (Goh et al. 2014) *Read below


*Use of the following functions is not recommended since their results differ from what is given in their original papers:
Please note it does not mean the models are incorrect**. However, we were unable to replicate them in this library and therefore would not recommend them.
  - lateralspreads/spt_based.py:
    - Baziar2013()
    - Goh()

**If you found bugs please report it to massoud.hosseinali@utah.edu

Install the following Python dependencies before using these codes:
 - numpy (http://www.numpy.org/)
 - pandas (https://pandas.pydata.org/)
 - sklearn (https://scikit-learn.org)
 - matplotlib (https://matplotlib.org/)
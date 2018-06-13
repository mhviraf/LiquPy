# **************************************************************************************
# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquePy/LiquePy
# **************************************************************************************

This open-source Python library is an attempt to facilitate research on soil liquefaction and lateral spreads by providing researchers/engineers with premade/verified Python codes.

If you are willing to contribute write an email to massoud.hosseinali@utah.edu

So far the following methods have been added:
  - lateralspreads/spt_based.py:
    - MLR (Youd, Hansen, & Bartlett 2002)
    - MLR (Bardet et al. 2002)
    - Genetic programming (Javadi et al. 2006)
    - Evolutionary-based approach (Rezania et al. 2011)
    - MARS (Goh et al. 2014), this method has not been yet verified and the results differs from what is given in the original paper


Install the following Python dependencies before using these codes:
 - numpy (http://www.numpy.org/)
 - pandas (https://pandas.pydata.org/)
 - sklearn (scikit-learn.org)
 - matplotlib (https://matplotlib.org/)
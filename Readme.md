# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
## https://github.com/LiquePy/LiquePy

This open-source Python library is an attempt to facilitate research on soil liquefaction and lateral spreads by providing researchers/engineers with premade/verified Python codes.


If you are willing to contribute or found bugs write an email to massoud.hosseinali@utah.edu


### What is included?
So far the following methods have been added:
  - under "liquefaction/spt_based.py":
    - Simplified factor of safety for triggering of soil liquefaction based on Idriss & Boulanger (2008)
  - under "lateralspreads/spt_based.py":
    - Multi Linear Regression (MLR) (Youd, Hansen, & Bartlett 2002)
    - Multi Linear Regression (MLR) (Bardet et al. 2002)
    - Genetic programming (Javadi et al. 2006)
    - Evolutionary-based approach (Rezania et al. 2011)
    - Artificial Neural Network & Genetic Algorithm (Baziar & Azizkani 2013) *Read below
    - Multivariate Adaptive Regression Splines (MARS) (Goh et al. 2014) *Read below


### What is not verified yet?
Use of the following functions is not recommended since their results differ from what is given in their original papers:
Please note it does not mean the models are incorrect. However, we were unable to replicate them in this library and therefore would not recommend them.
  - under "lateralspreads/spt_based.py":
    - Baziar2013()
    - Goh()


### Dependencies:
Install the following Python dependencies before using these codes:
 - numpy (http://www.numpy.org/)
 - pandas (https://pandas.pydata.org/)
 - sklearn (https://scikit-learn.org)
 - matplotlib (https://matplotlib.org/)


 ### References:
 - Bardet, J. P., Tobita, T., Mace, N., & Hu, J. (2002). Regional modeling of liquefaction-induced ground deformation. Earthquake Spectra, 18(1), 19-46.
 - Baziar, M. H., & Saeedi Azizkandi, A. (2013). Evaluation of lateral spreading utilizing artificial neural network and genetic programming. International Journal of Civil Engineering, (2), 100-111.
 - Goh, A. T., & Zhang, W. G. (2014). An improvement to MLR model for predicting liquefaction-induced lateral spread using multivariate adaptive regression splines. Engineering Geology, 170, 1-10.
 - Idriss, I. M., & Boulanger, R. W. (2008). Soil liquefaction during earthquakes. Earthquake Engineering Research Institute.
 - Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.
 - Rezania, M., Faramarzi, A., & Javadi, A. A. (2011). An evolutionary based approach for assessment of earthquake-induced soil liquefaction and lateral displacement. Engineering Applications of Artificial Intelligence, 24(1), 142-153.
 - Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.
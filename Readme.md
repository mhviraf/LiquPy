# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
## https://github.com/LiquPy/LiquPy

This open-source Python library is an attempt to facilitate research on soil liquefaction and lateral spreads by providing researchers/engineers with premade/verified Python codes.

### User installation:
The easiest way to install LiquPy is using `pip`:
```
pip install -U liqupy
```

or `conda`:
```
conda install liqupy
```

If you are willing to contribute or found bugs write an email to massoud.hosseinali@utah.edu


### What is included?
So far the following methods have been added:
  - under "boreholes.py":
    - Simplified liqufaction triggering analysis
       * CRR based on Boulanger and Idriss (2004) & Boulanger and Idriss (2008)
       * Adjustments for fines content from (1) Boulanger and Idriss (2004), and (2) Cetin et al. (2004)
       * Shear stress reduction factors available from (1) Golesorkhi 1989, (2) Idriss 1999, and (3) Liao & Whitman 1986
       * Magnitude scaling factor from Idriss (1999)
       * Overburden correction factor from Boulanger and Idriss (2004)
       * Triggering correlation of liquefaction in clean sands from Idriss and Boulanger (2004)
       * Probabilistic approaches from Cetin et al. (2004)
    - Lateral Displacement Index (LDI) and settlement (Zhang et al. 2004)
  - under "points.py":
    - Lateral spread analysis 
       * Multi Linear Regression (MLR) (Youd, Hansen, & Bartlett 2002)
       * Multi Linear Regression (MLR) (Bardet et al. 2002)
       * Genetic programming (Javadi et al. 2006)
       * Evolutionary-based approach (Rezania et al. 2011)
       * Artificial Neural Network & Genetic Algorithm (Baziar & Azizkani 2013) *Read below
       * Multivariate Adaptive Regression Splines (MARS) (Goh et al. 2014) *Read below

### What is not verified yet?
Use of the following functions of our library is not recommended since their results differ from what is given in their original papers; please note it does not mean these models are incorrect. However, we were unable to replicate them in this library and therefore would not recommend using these functions of our library:
  - under "points.py":
    - Lateral spread analysis 
       * Artificial Neural Network & Genetic Algorithm (Baziar & Azizkani 2013) 
       * Multivariate Adaptive Regression Splines (MARS) (Goh et al. 2014) 


### Dependencies:
Python (>= 3.5)

Also, if you are NOT installing LiquPy through `pip install LiquPy`,  install the following Python dependencies before using these codes:
 - numpy (http://www.numpy.org/)
 - pandas (https://pandas.pydata.org/)
 - sklearn (https://scikit-learn.org)
 - matplotlib (https://matplotlib.org/)


 ### References:
 - Bardet, J. P., Tobita, T., Mace, N., & Hu, J. (2002). Regional modeling of liquefaction-induced ground deformation. Earthquake Spectra, 18(1), 19-46.
 - Baziar, M. H., & Saeedi Azizkandi, A. (2013). Evaluation of lateral spreading utilizing artificial neural network and genetic programming. International Journal of Civil Engineering, (2), 100-111.
 - Boulanger, R. W., & Idriss, I. M. (2004). State normalization of penetration resistance and the effect of overburden stress on liquefaction resistance. Proceedings 11th SDEE and 3rd ICEGE, Uni of California, Berkeley, CA.
 - Cetin, K. O., Seed, R. B., Der Kiureghian, A., Tokimatsu, K., Harder Jr, L. F., Kayen, R. E., & Moss, R. E. (2004). Standard penetration test-based probabilistic and deterministic assessment of seismic soil liquefaction potential. Journal of geotechnical and geoenvironmental engineering, 130(12), 1314-1340.
 - Goh, A. T., & Zhang, W. G. (2014). An improvement to MLR model for predicting liquefaction-induced lateral spread using multivariate adaptive regression splines. Engineering Geology, 170, 1-10.
 - Golesorkhi, R. (1989). Factors influencing the computational determination of earthquake-induced shear stresses in sandy soils. University of California, Berkeley.
 - Idriss, I. M. (1999). An update to the Seed-Idriss simplified procedure for evaluating liquefaction potential. Proc., TRB Worshop on New Approaches to Liquefaction, Pubbl. n. FHWA-RD-99-165, Federal Highway Administation.
 - Idriss, I. M., & Boulanger, R. W. (2006). Semi-empirical procedures for evaluating liquefaction potential during earthquakes. Soil Dynamics and Earthquake Engineering, 26(2-4), 115-130.
 - Idriss, I. M., & Boulanger, R. W. (2008). Soil liquefaction during earthquakes. Earthquake Engineering Research Institute.
 - Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.
 - Liao, S. S., & Whitman, R. V. (1986). A catalog of liquefaction and non-liquefaction occurrences during earthquakes. Department of Civil Engineering, MIT.
 - Rezania, M., Faramarzi, A., & Javadi, A. A. (2011). An evolutionary based approach for assessment of earthquake-induced soil liquefaction and lateral displacement. Engineering Applications of Artificial Intelligence, 24(1), 142-153.
 - Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.
 - Zhang, G., Robertson, P. K., & Brachman, R. W. I. (2004). Estimating liquefaction-induced lateral displacements using the standard penetration test or cone penetration test. Journal of Geotechnical and Geoenvironmental Engineering, 130(8), 861-871.
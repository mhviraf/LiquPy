# **************************************************************************************
# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquePy/LiquePy
# **************************************************************************************

# parameters are defined based on the definitions given in Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.
# mode = 'f': Free-face & 's': sloping ground (not defined in Youd et al. 2002)
# R = R5the nearest horizontal or map distance from the site to the seismic energy source, in kilometers
# M = the moment magnitude of the earthquake
# T = the cumulative thickness of saturated granular layers with corrected blow counts, (N1)60, less than 15, in meters
# F = the average fines content ~fraction of sediment sample passing a No. 200 sieve! for granular materials included within T , in percent
# D = the average mean grain size for granular materials within T15, in millimeters
# S = the ground slope, in percent
# W = the free-face ratio defined as the height (H) of the free face divided by the distance (L) from the base of the free face to the point in question, in percent


# dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# loading datasets (cite Youd et al. 2002 if you are using this default dataset)
Data = pd.read_excel('YoudHansenBartlett2002.xls')
Data_FreeFace = Data.loc[Data.loc[:, 'b0'] == 1, :]   # free face points, 'f'
Data_Slope = Data.loc[Data.loc[:, 'b0'] == 0, :]      # sloping ground points. 's'


# MLR (Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.)
def Bartlett(mode, M, R, T, F, D, W, S):
    R0 = np.power(10, 0.89*M-5.64)
    Rstar = R + R0
    if mode == 'f':
        logDH = -16.713 + 1.532*M - 1.406*np.log10(Rstar) - 0.012*R + 0.592*np.log10(W) + 0.540*np.log10(T) + 3.413*np.log10(100-F) - 0.795*np.log10(D + 0.1)
    elif mode == 's':
        logDH = -16.213 + 1.532*M - 1.406*np.log10(Rstar) - 0.012*R + 0.338*np.log10(S) + 0.540*np.log10(T) + 3.413*np.log10(100-F) - 0.795*np.log10(D + 0.1)
    return np.power(10, logDH)


# Bardet et al 2002 (Bardet, J. P., Tobita, T., Mace, N., & Hu, J. (2002). Regional modeling of liquefaction-induced ground deformation. Earthquake Spectra, 18(1), 19-46.)
def Bardet(mode, M, R, T, F, D, W, S):
    if mode == 'f':
        logDH = -6.815 - 0.465 + 1.017 * M - 0.278 * np.log10(R) - 0.026 * R + 0.497*np.log10(W) + 0.558 * np.log10(T)
    elif mode == 's':
        logDH = -6.815 + 1.017*M - 0.278*np.log10(R) - 0.026*R + 0.454*np.log10(S) + 0.558*np.log10(T)
    return np.power(10, logDH)


# Javadi et al 2006 (Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.)
def Javadi(mode, M, R, T, F, D, W, S):
    if F == 0: F += 0.001  # Correction for F = 0 | this model does not capture F = 0
    if mode == 'f':
        DH = (-163.1/M**2 + 57/R/F - 0.0035*T**2/W/D**2 + 0.02*T**2/F/D**2 - 0.26*(T/F)**2 + 0.006*T**2
              - 0.0013*W**2 + 0.0002*M**2*W*T + 3.7)
    elif mode == 's':
        DH = (-0.8*F/M + 0.0014*F**2 + 0.16*T + 0.112*S + 0.04*S*T/D - 0.026*R*D + 1.14)
    return DH


# Javadi et al 2006 - moderate (Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.)
def Javadi_moderate(mode, M, R, T, F, D, W, S):
    if mode == 'f':
        DH = (-234.1/(M**2*R*W) - 156/M**2 - 0.008*F/R**2/T + 0.01*W*T/R - 2.9/F - 0.036*M*T**2*D**2/R**2/W
            + 9.4*M/R/F - 4*10**-6*M*R**2/D + 3.84)
    elif mode == 's':
        DH = (-0.027*T**2*F/M**2 + 0.05*R*T/M**2/D + 0.44/M/R**2/S/T - 0.03*R -0.02*M/S/T - 5*10**-5*M*R/D**2
            + 0.075*M**2 - 2.4)
    return DH


# Rezania et al 2011 (Rezania, M., Faramarzi, A., & Javadi, A. A. (2011). An evolutionary based approach for assessment of earthquake-induced soil liquefaction and lateral displacement. Engineering Applications of Artificial Intelligence, 24(1), 142-153.)
def Rezania(mode, M, R, T, F, D, W, S):
    if F == 0: F += 0.001  # Correction for F = 0 | this model does not capture F = 0
    if mode == 'f':
        DH = (-2.1414*R**0.5*W**0.5/M**2/D**0.5 - 0.061863*T*F/np.sqrt(M*W) - 11.1201*M**2/R/W**0.5/F
              + 0.0017573*M**2*W**0.5*T/F**0.5/D + 1.9671)
    elif mode == 's':
        DH = (-1.6941*T**0.5*F**0.5/M**2/D**0.5 - 0.78905*R**0.5*S**0.5*T*F**0.5/M**2
              -2.2542*10**-12*M**0.5*T**2*D**2/R**0.5/S**0.5/F**2 + 0.036036*M*S**0.5*T/D**0.5 + 0.85441)
    return DH


# The following method has not yet been verified and the results differ from the ones given in the original paper
# Goh et al 2014 (Goh, A. T., & Zhang, W. G. (2014). An improvement to MLR model for predicting liquefaction-induced lateral spread using multivariate adaptive regression splines. Engineering Geology, 170, 1-10.)
def Goh(mode, M, R, T, F, D, W, S):
    if mode == 'f':
        BF1 = max(0., np.log10(T) - 0.699)
        BF2 = max(0., 0.699 - np.log10(T))
        BF3 = max(0., M - 6.6)
        BF4 = max(0., R - 6)
        BF5 = max(0., 6 - R)
        BF6 = max(0., 1.9315 - np.log10(100-F))
        BF7 = max(0., 1.2495 - np.log10(W))
        BF8 = max(0., -0.2441 - np.log10(D+0.1))
        BF9 = BF7*max(0., M - 6.8)
        BF10 = BF7*max(0., 6.8 - M)
        BF11 = BF7*max(0., np.log10(T) - 1.1038)
        BF12 = BF7*max(0., 1.1038 - np.log10(T))
        BF13 = BF6*max(0., 0.906 - np.log10(W))
        BF14 = max(0., 0.9823 - np.log10(T))
        BF15 = BF8*max(0, R-21)
        BF16 = BF5*max(0., 0.5682 - np.log10(T))
        BF17 = BF1*max(0., np.log(100-F)-1.8808)
        logDH = (0.464 - 2.022*BF1 + 3.456*BF2 + .919*BF3 - 0.027*BF4 + 0.270*BF5 - 6.056*BF6 - 1.477*BF7
                 + 1.555*BF8 + .867*BF9 - 2.3*BF10 + 8.488*BF11 + .597*BF12 + 6.796*BF13 - 3.508*BF14 + 0.071*BF15
                 - 0.116*BF16 + 16.577*BF17)
    elif mode == 's':
        R0 = np.power(10, 0.89 * M - 5.64)
        Rstar = R + R0
        BF1 = max(0., 7.5-M)
        BF2 = BF1*max(0, np.log10(Rstar) - 0.8873)
        BF3 = BF1*max(0, 0.8873 - np.log10(Rstar))
        BF4 = max(0, R-27)
        BF5 = max(0, 27-R)
        BF6 = BF5*max(0, np.log10(T)-0.9445)
        BF7 = BF5*max(0, 0.9445 - np.log10(T))
        BF8 = max(0, np.log10(S) + 0.6198)*max(0, np.log10(T) - 1.0294)
        BF9 = max(0., np.log10(S) + 0.6198)*max(0., 1.0294 - np.log10(T))
        BF10 = max(0., np.log10(D+0.1) + 0.4763)
        BF11 = max(0., -0.4763 - np.log10(D + 0.1))
        BF12 = max(0., np.log10(T) - 0.7404)
        BF13 = max(0., 0.7404 - np.log10(T))
        BF14 = BF1*max(0, np.log10(100-F) - 1.8808)
        BF15 = max(0, 1.9912 - np.log10(100-F))
        BF16 = max(0., np.log10(S) + 0.6198)*max(0., 1.9868 - BF6)
        BF17 = BF15*max(0., np.log10(S) + 0.2518)
        BF18 = max(0., M - 6.6)
        logDH = (-1.766 - 1.647*BF1 + 3.102*BF2 + 1.78*BF3 - 0.035*BF4 +0.08*BF5 + 0.798*BF6 -.036*BF7
                 -13.161*BF8 + .52*BF9 - .658*BF10 - 3.312*BF11 - 0.976*BF12 - 0.662*BF13 + 35.986*BF14
                 -3.357*BF15 + 18.876*BF16 - 17.095*BF17 + 1.864*BF18)
    return np.power(10, logDH)


verification_figures = 998


# Draws a plot of the predicted values vs. measured values of the method
def verify(method):
    print(str(method))
    global verification_figures
    verification_figures += 2
    plt.figure(verification_figures)  # free face ************************
    plt.subplot(1, 2, 1)
    x = []
    y = []
    MSE = 0
    for i in range(len(Data_FreeFace)):
        x.append(Data_FreeFace.iloc[i, 1])
        args = ('f',)
        args = args + tuple(Data_FreeFace.iloc[i, [3, 4, 5, 6, 7, 8]].values)
        args = args + (0,)
        Yi = method(*args)
        MSE += (Data_FreeFace.iloc[i, 1] - Yi)**2/(len(Data_FreeFace))
        y.append(Yi)
    plt.scatter(x, y)
    plt.xlabel('measured (m)')
    plt.ylabel('predicted (m)')
    if min(y) > 0: plt.xlim(left=0)
    plt.ylim(plt.xlim())
    plt.title('Free face')
    y2 = y
    print('Free face:')
    print('Data length = {}'.format(len(y)))
    print('MSE = {}; RMSE = {}; r2= {}'.format(MSE, np.sqrt(MSE/len(Data_FreeFace)), r2_score(x, y)))

    # residuals
    plt.figure(verification_figures+1)
    plt.subplot(1, 2, 1)
    plt.scatter(x, np.subtract(y, x))
    plt.title('Free face')
    plt.xlabel('Measured displacement')
    plt.ylabel('Residuals')
    plt.ylim([-5, 5])

    plt.figure(verification_figures)
    plt.subplot(1, 2, 2) # gently sloping ************************
    x = []
    y = []
    MSE = 0
    for i in range(len(Data_Slope)):
        x.append(Data_Slope.iloc[i, 1])
        args = ('s',)
        args = args + tuple(Data_Slope.iloc[i, [3, 4, 5, 6, 7, 8, 9]].values)
        Yi = method(*args)
        MSE += (Data_Slope.iloc[i, 1] - Yi) ** 2 / (len(Data_Slope))
        y.append(Yi)
    plt.scatter(x, y)
    plt.xlabel('measured (m)')
    plt.ylabel('predicted (m)')
    if min(y) > 0: plt.xlim(left=0)
    plt.ylim(plt.xlim())
    plt.title('Sloping ground')
    print('Sloping ground:')
    print('Data length = {}'.format(len(y)))
    print('MSE = {}; RMSE = {}; r2= {}'.format(MSE, np.sqrt(MSE / len(Data_FreeFace)), r2_score(x, y)))
    # plt.figure()
    # [y2.append(yi) for yi in y]
    # plt.hist(np.log10(y2))

    # residuals
    plt.figure(verification_figures+1)
    plt.subplot(1, 2, 2)
    plt.scatter(x, np.subtract(y, x))
    plt.title('Sloping ground')
    plt.xlabel('Measured displacement')
    plt.ylabel('Residuals')
    plt.ylim([-5, 5])



# an example of how to get horizontal ground displacement predictions from the Bartlett's MLR model:
print(Bartlett('f', 7.217982, 18.385526, 8.567101, 17.115035, 0.359680, 10.656302, 0))  # returns predicted values of Bartlett's MLR method at a single point
verify(Bartlett)  # plots predicted vs. measured displacements of Bartlett's method
plt.show()  # shows the plots



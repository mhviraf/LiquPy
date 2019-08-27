# **************************************************************************************
# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquPy/LiquPy
# **************************************************************************************

# parameters are defined based on the definitions given in Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.
# mode = 'f': Free-face & 's': sloping ground (not defined in Youd et al. 2002)
# R = the horizontal or mapped distance from the site in question to the nearest bound of the seismic energy source, in kilometers
# M = the moment magnitude of the earthquake
# T = the cumulative thickness of saturated granular layers with corrected blow counts, (N1)60, less than 15, in meters
# F = the average fines content (fraction of sediment sample passing a No. 200 sieve) for granular materials included within T , in percent
# D = the average mean grain size for granular materials within T15, in millimeters
# S = the ground slope, in percent
# W = the free-face ratio defined as the height (H) of the free face divided by the distance (L) from the base of the free face to the point in question, in percent
# b0 = 1 for free face and 0 for sloping ground data points


__all__ = ['calc_ls_bartlett',
           'calc_ls_bardet',
           'calc_ls_javadi2006',
           'calc_ls_javadi_moderate2006',
           'calc_ls_rezania',
           'calc_ls_baziar2013',
           'calc_ls_goh',
           'verify_ls_model']

# dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Empirical methods ************************************************************


# MLR (Youd, T. L., Hansen, C. M., & Bartlett, S. F. (2002). Revised multilinear regression equations for prediction of lateral spread displacement. Journal of Geotechnical and Geoenvironmental Engineering, 128(12), 1007-1017.)
def calc_ls_bartlett(mode, M, R, T, F, D, W, S):
    if (M < 6) or (M > 8):
        warnings.warn('Value of M not in recommended range by Youd et al. (2002): 6 < M < 8')
    if (W < 1) or (W > 20):
        warnings.warn('Value of W not in recommended range by Youd et al. (2002): 1 < W(%) < 20')
    if (S < 0.1) or (S > 6):
        warnings.warn('Value of S not in recommended range by Youd et al. (2002): 0.1 < S(%) < 6')
    if (T < 1) or (T > 15):
        warnings.warn('Value of T not in recommended range by Youd et al. (2002): 1 < T (m) < 15')
    if R < 0.5:
        R = 0.5
        
    R0 = np.power(10, 0.89*M-5.64)
    Rstar = R + R0
    if mode == 'f':
        logDH = -16.713 + 1.532*M - 1.406*np.log10(Rstar) - 0.012*R + 0.592*np.log10(W) + 0.540*np.log10(T) + 3.413*np.log10(100-F) - 0.795*np.log10(D + 0.1)
    elif mode == 's':
        logDH = -16.213 + 1.532*M - 1.406*np.log10(Rstar) - 0.012*R + 0.338*np.log10(S) + 0.540*np.log10(T) + 3.413*np.log10(100-F) - 0.795*np.log10(D + 0.1)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')
    
    Dh = np.power(10, logDH)
    if Dh > 6:
        warnings.warn('Flow failure may be possible and Youd et al. (2002) may not be applicable')
    return Dh


# Bardet et al 2002 (Bardet, J. P., Tobita, T., Mace, N., & Hu, J. (2002). Regional modeling of liquefaction-induced ground deformation. Earthquake Spectra, 18(1), 19-46.)
def calc_ls_bardet(mode, M, R, T, F, D, W, S):
    if mode == 'f':
        logDH = -6.815 - 0.465 + 1.017 * M - 0.278 * np.log10(R) - 0.026 * R + 0.497*np.log10(W) + 0.558 * np.log10(T)
    elif mode == 's':
        logDH = -6.815 + 1.017*M - 0.278*np.log10(R) - 0.026*R + 0.454*np.log10(S) + 0.558*np.log10(T)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')
    
    return np.power(10, logDH)


# Javadi et al 2006 (Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.)
def calc_ls_javadi2006(mode, M, R, T, F, D, W, S):
    if F == 0: F += 0.001  # Correction for F = 0 | this model does not capture F = 0
    if mode == 'f':
        DH = (-163.1/M**2 + 57/R/F - 0.0035*T**2/W/D**2
              + 0.02*T**2/F/D**2 - 0.26*T**2/F**2 + 0.006*T**2
              - 0.0013*W**2 + 0.0002*M**2*W*T + 3.7)
    elif mode == 's':
        DH = (-0.8*F/M + 0.0014*F**2 + 0.16*T + 0.112*S
              + 0.04*S*T/D - 0.026*R*D + 1.14)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')

    if DH <= 1.5:
        warnings.warn('Predicted displacement was greater than 1.5 meters. The GP model for moderate displacements are being used')
        DH = calc_ls_javadi_moderate2006(mode=mode, M=M, R=R, T=T, F=F, D=D, W=W, S=S)
    
    return DH


# Javadi et al 2006 - moderate (Javadi, A. A., Rezania, M., & Nezhad, M. M. (2006). Evaluation of liquefaction induced lateral displacements using genetic programming. Computers and Geotechnics, 33(4-5), 222-233.)
def calc_ls_javadi_moderate2006(mode, M, R, T, F, D, W, S):
    # you may either use this function to directly use moderate models
    if mode == 'f':
        DH = (-234.1/(M**2*R*W) - 156/M**2 - 0.008*F/R**2/T + 0.01*W*T/R - 2.9/F - 0.036*M*T**2*D**2/R**2/W
            + 9.4*M/R/F - 4*10**-6*M*R**2/D + 3.84)
    elif mode == 's':
        DH = (-0.027*T**2*F/M**2 + 0.05*R*T/M**2/D + 0.44/M/R**2/S/T - 0.03*R -0.02*M/S/T - 5*10**-5*M*R/D**2
            + 0.075*M**2 - 2.4)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')
    
    return DH


# Rezania et al 2011 (Rezania, M., Faramarzi, A., & Javadi, A. A. (2011). An evolutionary based approach for assessment of earthquake-induced soil liquefaction and lateral displacement. Engineering Applications of Artificial Intelligence, 24(1), 142-153.)
def calc_ls_rezania2011(mode, M, R, T, F, D, W, S):
    if F == 0: F += 0.001  # Correction for F = 0 | this model does not capture F = 0
    if mode == 'f':
        DH = (-2.1414*R**0.5*W**0.5/M**2/D**0.5 - 0.061863*T*F/M**0.5/W**0.5 - 11.1201*M**2/R/W**0.5/F
              + 0.0017573*M**2*W**0.5*T/F**0.5/D + 1.9671)
    elif mode == 's':
        DH = (-1.6941*T**0.5*F**0.5/M**2/D**0.5 - 0.78905*R**0.5*S**0.5*T*F**0.5/M**2
              -2.2542*10**-12*M**0.5*T**2*D**2/R**0.5/S**0.5/F**2 + 0.036036*M*S**0.5*T/D**0.5 + 0.85441)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')

    if DH < 0:
        DH = 0
        
    return DH


def calc_ls_rezania_moderate2011(mode, M, R, T, F, D, W, S):
    # you may either use this function to directly use moderate models
    if F == 0: F += 0.001  # Correction for F = 0 | this model does not capture F = 0
    if mode == 'f':
        DH = (-12.7493*T*D**0.5/M**2/R**0.5/W**0.5 - 3.4311*F/M**2/R**0.5 - 24.0261/M/R**0.5/T**0.5/F/D
              -355.8433/M/W**0.5 + 4.0048E-6*R**0.5*F**2/M/W**0.5/D**2 + 128.522/M**.5/W**.5
              -5.3745E-4*M**2*R**0.5*W**.5 + 0.95605)
    elif mode == 's':
        DH = (-1.8017*R*D**0.5/M**2 - 0.12148*F/M - 31.5315/M**0.5
              +2.9385*M**0.5/S**0.5/T + 1.9056E-4*M**2*S**0.5*T*F**0.5*D**0.5/R**2
              +13.3224)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')
    
    if DH < 0:
        DH = 0
        
    return DH


# Baziar and Azizkani 2013 (Baziar, M. H., & Saeedi Azizkandi, A. (2013). Evaluation of lateral spreading utilizing artificial neural network and genetic programming. International Journal of Civil Engineering, (2), 100-111.)
def calc_ls_baziar2013(mode, M, R, T, F, D, W, S):
    DH = (54.36*T/(D+0.6532) - 55.34*T/(D+0.6689) + 196.9*T/(W+0.9212) - 199.8*T/(W+0.9434) + 0.0446*(W-S)/R
          - 1.718/(S+0.8956) - 0.02452*T*F - 0.00625*F*S + 0.001474*R*(W-T) - 0.06875*T*(W-S)
          + M*(0.1058*T + 0.009652*T*W - 0.1225) + 0.00024*T*F**2 - 0.00255*R*W*S + 2.6)
    if DH < 0: DH = 0
    return DH


# Goh et al 2014 (Goh, A. T., & Zhang, W. G. (2014). An improvement to MLR model for predicting liquefaction-induced lateral spread using multivariate adaptive regression splines. Engineering Geology, 170, 1-10.)
def calc_ls_goh(mode, M, R, T, F, D, W, S):
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
        logDH = (-1.766 - 1.647*BF1 + 3.102*BF2 + 1.78*BF3 - 0.035*BF4 + 0.08*BF5 + 0.798*BF6 - .036*BF7
                 - 13.161*BF8 + .52*BF9 - .658*BF10 - 3.312*BF11 - 0.976*BF12 - 0.662*BF13 + 35.986*BF14
                 - 3.357*BF15 + 18.876*BF16 - 17.095*BF17 + 1.864*BF18)
    else:
        raise ValueError('The mode could be either f for free face or s for sloping ground')
    
    return np.power(10, logDH)

verification_figures = 998


# Draws a plot of the predicted values vs. measured values of the method
def verify_ls_model(ls_calculation_method, data):
    # NOTE: only point-based ls_calc models could be passed as ls_calculation_method
    data_FreeFace = data.loc[data.loc[:, 'b0'] == 1, :]  # free face points, 'f'
    data_Slope = data.loc[data.loc[:, 'b0'] == 0, :]  # sloping ground points. 's'

    print(str(ls_calculation_method))
    global verification_figures
    verification_figures += 2
    plt.figure(verification_figures)  # free face ************************
    plt.subplot(1, 2, 1)
    x = []
    y = []
    MSE = 0
    for i in range(len(data_FreeFace)):
        x.append(data_FreeFace.iloc[i, 1])
        args = ('f',)
        args = args + tuple(data_FreeFace.iloc[i, [3, 4, 5, 6, 7, 8]].values)
        args = args + tuple([0]) # setting S = 0 in this case
        Yi = ls_calculation_method(*args)
        MSE += (data_FreeFace.iloc[i, 1] - Yi)**2/(len(data_FreeFace))
        y.append(Yi)
    plt.scatter(x, y)
    plt.xlabel('measured (m)')
    plt.ylabel('predicted (m)')

    if min(y) > 0:
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
    if plt.ylim()[1] > plt.xlim()[1]:
        pltlim = plt.ylim()
    else:
        pltlim = plt.xlim()
    plt.xlim(pltlim)
    plt.ylim(pltlim)
    plt.plot([pltlim[0], pltlim[1]], [pltlim[0], pltlim[1]], 'g--')
    plt.plot([pltlim[0], pltlim[1]], [pltlim[0], pltlim[1]/2], 'g--')
    plt.plot([pltlim[0], pltlim[1]/2], [pltlim[0], pltlim[1]], 'g--')
    plt.title('Free face')
    y2 = y
    print('Free face:')
    print('Data length = {}'.format(len(y)))
    print('MSE = {}; RMSE = {}; r2= {}'.format(MSE, np.sqrt(MSE/len(data_FreeFace)), r2_score(x, y)))
    
    
    # residuals
    plt.figure(verification_figures+1)
    plt.subplot(1, 2, 1)
    plt.scatter(x, np.subtract(y, x))
    plt.plot(pltlim, [0, 0], 'g--')
    plt.title('Free face')
    plt.xlabel('Measured displacement')
    plt.ylabel('Residuals')
    plt.ylim([-5, 5])

    plt.figure(verification_figures)
    plt.subplot(1, 2, 2) # gently sloping ************************
    x = []
    y = []
    MSE = 0
    for i in range(len(data_Slope)):
        x.append(data_Slope.iloc[i, 1])
        args = ('s',)
        args = args + tuple(data_Slope.iloc[i, [3, 4, 5, 6, 7]].values)
        args = args + tuple([0])  # setting W = 0
        args = args + tuple(data_Slope.iloc[i, [9]].values)
        Yi = ls_calculation_method(*args)
        MSE += (data_Slope.iloc[i, 1] - Yi) ** 2 / (len(data_Slope))
        y.append(Yi)
    plt.scatter(x, y)
    plt.xlabel('measured (m)')

    if min(y) > 0:
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
    if plt.ylim()[1] > plt.xlim()[1]:
        pltlim = plt.ylim()
    else:
        pltlim = plt.xlim()
    plt.xlim(pltlim)
    plt.ylim(pltlim)
    plt.plot([pltlim[0], pltlim[1]], [pltlim[0], pltlim[1]], 'g--')
    plt.plot([pltlim[0], pltlim[1]], [pltlim[0], pltlim[1]/2], 'g--')
    plt.plot([pltlim[0], pltlim[1]/2], [pltlim[0], pltlim[1]], 'g--')
    plt.title('Sloping ground')
    plt.tight_layout()
    print('Sloping ground:')
    print('Data length = {}'.format(len(y)))
    print('MSE = {}; RMSE = {}; r2= {}'.format(MSE, np.sqrt(MSE / len(data_FreeFace)), r2_score(x, y)))
    # plt.figure()
    # [y2.append(yi) for yi in y]
    # plt.hist(np.log10(y2))

    # residuals
    plt.figure(verification_figures+1)
    plt.subplot(1, 2, 2)
    plt.scatter(x, np.subtract(y, x))
    plt.plot(pltlim, [0, 0], 'g--')
    plt.title('Sloping ground')
    plt.xlabel('Measured displacement')
    plt.ylim([-5, 5])
    plt.tight_layout()


if __name__=='__main__':
    # loading demo dataset from Youd et al. (2002)
    default_dataset = pd.read_excel('default_datasets/YoudHansenBartlett2002_demo.xls')

    # an example of how to get horizontal ground displacement predictions from the Bartlett's MLR model:
    # a) running on a single point
    print(calc_ls_bartlett(mode='f', M=7.217982, R=18.385526, T=8.567101, F=17.115035, D=0.359680, W=10.656302, S=0))  # returns predicted values of Bartlett's MLR method at a single point

    # b) running on a database
    verify_ls_model(calc_ls_baziar2013, default_dataset)  # plots predicted vs. measured displacements + residuals of Bartlett's method on YoudHansenBartlett2002_demo dataset
    plt.show()  # shows the plots


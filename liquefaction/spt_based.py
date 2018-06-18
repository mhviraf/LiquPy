# **************************************************************************************
# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquePy/LiquePy
# **************************************************************************************

import numpy as np
import pandas as pd


spt_idriss_boulanger_bore_data_appendix_a = pd.read_excel('spt_Idriss_Boulanger.xlsx')


def simplified_liquefaction_triggering_fos(borelog, Pa, M, Zw, sampler_correction_factor,
                                          liner_correction_factor, hammer_energy, rod_extension):
    # Pa = Peak ground acceleration (g)
    # M = Earthquake magnitude
    # Zw = water table depth (m)
    output = []
    sigmavp = 0
    sigmav = 0
    depth = 0
    hydro_pore_pressure = 0
    gamma = borelog.iloc[0, 6]
    for i, row in borelog.iterrows():
        rod_length = row[1] + rod_extension
        Nspt = row[2]
        ce = hammer_energy / 60
        if rod_length < 3:
            cr = 0.75
        elif rod_length < 4:
            cr = 0.8
        elif rod_length < 6:
            cr = 0.85
        elif rod_length < 10:
            cr = 0.95
        else:
            cr = 1
        cs = sampler_correction_factor
        cb = liner_correction_factor
        N60 = Nspt * ce * cr * cs * cb

        sigmav = sigmav + (row[1] - depth)*0.5*(row[6]+gamma)
        if row[1] > Zw:
            hydro_pore_pressure = (row[1]-Zw) * 9.81
        sigmavp = sigmav - hydro_pore_pressure

        if row[4] == 1:
            N60 = 'n.a.'
            N160 = 'n.a.'
            N160cs = 'n.a.'
        else:
            if sigmavp == 0:
                CN = 1
            else:
                CN = min(1.7, np.sqrt(100 / sigmavp))

            N160 = CN*N60  #  (N1)60 proposed by Liao and Whitman (1986)

            delta_n = np.exp(1.63 + 9.7/(row[5]+0.01) - (15.7/(row[5]+0.01))**2)
            N160cs = N160 + delta_n

        rd = np.exp((-1.012-1.126*np.sin(row[1]/11.73+5.133)) + (0.106+0.118*np.sin(row[1]/11.28+5.142))*M)

        CSR = 0.65*sigmav/sigmavp*Pa*rd

        if row[4] == 1 or row[1] < Zw:
            CRR0 = 'n.a.'
            CRR = 'n.a.'
            FoS = 'n.a.'
            MSF = 'n.a'
            k_sigma = 'n.a.'
        else:
            MSF = min(1.8, 6.9 * np.exp(-M / 4) - 0.058)
            k_sigma = min(1.1, 1 - (min(0.3, 1 / (18.9 - 2.55 * np.sqrt(N160cs)))) * np.log(sigmavp / 100))

            if N160cs < 37.5:
                CRR0 = np.exp(N160cs / 14.1 + (N160cs / 126) ** 2 - (N160cs / 23.6) ** 3 + (N160cs / 25.4) ** 4 - 2.8)
            else:
                CRR0 = 2
            CRR = min(2., CRR0*MSF*k_sigma)
            if CRR/CSR > 2:
                FoS = 2
            else:
                FoS = CRR/CSR


        depth = row[1]
        gamma = row[6]

        output.append([row[1], ce, cb, cr, cs, N60, sigmav, sigmavp, CN, N160, delta_n, N160cs, rd, CSR, MSF, CRR0, CRR, FoS])

    pd.DataFrame(output).to_excel('spt_based_fos_outputs.xls',
                                  header= ['depth', 'ce', 'cb', 'cr', 'cs', 'N60', 'sigmav', 'sigmavp', 'CN', 'N160', 'delta_n', 'N160cs', 'rd', 'CSR', 'MSF', 'CRR0', 'CRR', 'FS'])
    print('spt_based_fos_outputs.xls has been saved.')

# verification with Example of SPT-based liquefaction triggering analysis for a single boring (Soil Liquefaction During Earthquake textbook by Idriss and Boulanger)
simplified_liquefaction_triggering_fos(spt_idriss_boulanger_bore_data_appendix_a, 0.280, 6.9, 1.8, 1, 1, 75, 1.5)




# **************************************************************************************
# LiquePy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquePy/LiquePy
# **************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simplified_liquefaction_triggering_fos(borelog, Pa, M, Zw, sampler_correction_factor,
                                          liner_correction_factor, hammer_energy, rod_extension, save_to_file=False,
                                           visualize=False):
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

        output.append([row[1], ce, cb, cr, cs, N60, sigmav, sigmavp, CN, N160, delta_n, N160cs, rd, CSR, MSF, k_sigma, CRR0, CRR, FoS])

    output_dataframe = pd.DataFrame(output, columns=['depth', 'ce', 'cb', 'cr', 'cs', 'N60', 'sigmav', 'sigmavp', 'CN', 'N160', 'delta_n', 'N160cs', 'rd', 'CSR', 'MSF', 'k_simga', 'CRR0', 'CRR', 'FS'])

    # visualization of the liquefaction analysis
    if visualize:
        # subplot of SPT blow counts
        fig, ax = plt.subplots(ncols=2, figsize=(6, 6))
        ax[0].xaxis.tick_top()
        ax[0].xaxis.set_label_position('top')
        ax[1].xaxis.tick_top()
        ax[1].xaxis.set_label_position('top')

        soil_type_0 = ''
        depth_0 = 0
        spt_plot_max_x = max(output_dataframe.loc[output_dataframe.loc[:, 'N160'] != 'n.a.', 'N160'])*1.05
        for i, row in borelog.iterrows():
            soil_type_1 = row[3]
            depth_1 = -row[1]
            ax[0].text(spt_plot_max_x*.05, depth_1, soil_type_1, color=(0, 0, 0, 0.4), verticalalignment='center')

            if not soil_type_0 == soil_type_1:
                ax[0].plot([0, spt_plot_max_x], [(depth_1+depth_0)*.5, (depth_1+depth_0)*.5], color=(0, 0, 0, 0.15))

            ax[0].plot([0, spt_plot_max_x], [depth_1, depth_1], '--', color=(0, 0, 0, 0.05))

            depth_0 = depth_1
            soil_type_0 = soil_type_1

        ax[0].scatter(borelog.iloc[:, 2], -borelog.iloc[:, 1], marker='x', label='$N_{SPT}$')
        ax[0].scatter(output_dataframe.loc[output_dataframe.loc[:, 'N160'] != 'n.a.', 'N160'],
                    -borelog.ix[output_dataframe.loc[:, 'N160'] != 'n.a.', 1], marker='+', s=75, label='$(N_1)_{60}$')
        ax[0].legend(loc='lower right')
        ax[0].set(xlabel='SPT BLOW COUNT', ylabel='DEPTH', xlim=[0, spt_plot_max_x])
        ax[0].set_ylim(top=0)

        # subplot of CSR & CRR
        depth_0 = 0
        liquefiable_0 = False
        csr_0 = 0
        crr_0 = 0
        na_0 = False
        csrcrr_plot_max_x = 1

        for i, row in output_dataframe.iterrows():
            depth_1 = -row[0]
            ax[1].plot([-1, csrcrr_plot_max_x], [depth_1, depth_1], '--', color=(0, 0, 0, 0.05))
            na_1 = False
            csr_1 = row['CSR']
            crr_1 = row['CRR']
            if row['FS'] == 'n.a.':
                na_1 = True
                liquefiable_1 = False
                ax[1].text(-0.95, depth_1, 'Nonliquefied (Excluded)', color=(0, 0, 0, 0.4), verticalalignment='center')
            elif row['FS'] > 1:
                liquefiable_1 = False
                ax[1].text(-0.95, depth_1, 'Nonliquefied (FS = {} > 1.0)'.format(round(row['FS'], 1)),
                           color=(0, 0, 0, 0.4), verticalalignment='center')
            else:
                liquefiable_1 = True
                ax[1].text(-0.95, depth_1, 'Liquified (FS = {} < 1.0)'.format(round(row['FS'], 1)),
                           color=(0, 0, 0, 0.4),
                           verticalalignment='center')
            if i > 0:
                if not na_1 and not na_0:
                    ax[1].plot([csr_0, csr_1], [depth_0, depth_1], 'k--')
                    ax[1].plot([crr_0, crr_1], [depth_0, depth_1], 'k-')

            if not liquefiable_0 == liquefiable_1:
                ax[1].plot([-1, csrcrr_plot_max_x], [(depth_1+depth_0)*.5, (depth_1+depth_0)*.5], color=(0, 0, 0, 0.15))

            liquefiable_0 = liquefiable_1
            depth_0 = depth_1
            csr_0 = csr_1
            crr_0 = crr_1
            na_0 = na_1

        ax[1].plot([0, 0], [0, 0], 'k--', label='CSR')
        ax[1].plot([0, 0], [0, 0], 'k-', label='CRR')
        ax[1].legend(loc='lower right')
        ax[1].set(xlabel='CSR & CRR', ylabel='DEPTH', xlim=[-1, csrcrr_plot_max_x])
        ax[1].set_ylim(top=0)
        plt.xticks([0, 0.5, 1], [0, 0.5, 1])
        plt.show()

    if save_to_file:
        output_dataframe.to_excel('spt_based_fos_outputs.xls')
        print('spt_based_fos_outputs.xls has been saved.')
    else:
        return output_dataframe


if __name__ == '__main__':
    spt_idriss_boulanger_bore_data_appendix_a = pd.read_excel('spt_Idriss_Boulanger.xlsx')

    # verification with Example of SPT-based liquefaction triggering analysis for a single boring (Soil Liquefaction During Earthquake textbook by Idriss and Boulanger)
    simplified_liquefaction_triggering_fos(spt_idriss_boulanger_bore_data_appendix_a, 0.280, 6.9, 1.8, 1, 1, 75, 1.5, save_to_file=False, visualize=True)




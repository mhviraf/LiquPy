# **************************************************************************************
# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquPy/LiquPy
# **************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# borehole object
class Borehole:

    number_of_holes = 0

    # customize visualization
    viz_liquefied_text_kwargs = {'color': (0, 0, 0, 0.4), 'horizontalalignment': 'center', 'verticalalignment': 'center'}
    viz_dashed_guidelines = {'color': (0, 0, 1, 0.05), 'ls': '--'}

    def __init__(self, bore_log_data, name=None, units='metric'):
        Borehole.number_of_holes += 1

        self.bore_log_data = bore_log_data
        self.name = name
        
        if units == 'metric':
            self.units_length = 'm'
            self.units_area = '$m^2$'
        elif units == 'british':
            self.units_length = 'ft'
            self.units_area = '$ft^2$'

    def __del__(self):
        Borehole.number_of_holes -= 1

    # simplified liquefaction triggering analysis - stress-based
    def simplified_liquefaction_triggering_fos(self, Pa, M, Zw, sampler_correction_factor,
                                               liner_correction_factor, hammer_energy, rod_extension, rd_method='Idriss1999', fc_method = 'BI2004'):
        self.Pa = Pa  # Peak ground acceleration (g)
        self.M = M  # Earthquake magnitude
        self.Zw = Zw  # water table depth (in self.units_length units)
        self.sampler_correction_factor = sampler_correction_factor
        self.liner_correction_factor = liner_correction_factor
        self.hammer_energy = hammer_energy
        self.rod_extension = rod_extension
        self.rd_method= rd_method
        self.fc_method = fc_method

        output = []
        sigmavp = 0
        sigmav = 0
        depth = 0
        hydro_pore_pressure = 0
        gamma = self.bore_log_data.iloc[0, 6]
        for i, row in self.bore_log_data.iterrows():
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

                N160 = CN*N60  #  use of (N1)60 proposed by Liao and Whitman (1986)


                # Adjustments for fines content
                if self.fc_method = 'BI2004':
                    # Boulanger & Idriss (2004), default
                    delta_n = np.exp(1.63 + 9.7/(row[5]+0.01) - (15.7/(row[5]+0.01))**2)
                    N160cs = N160 + delta_n
                    
                elif self.fc_method = 'cetin2004':
                    # Cetin et al. (2004)
                    if row[5] > 5 or row[5] < 35:
                        warnings.warn('Cetin et al 2004 method of adjustments for fines content is only applicable to fines content in the range of 5% to 35%')
                    c_fines = 1 + 0.004*row[5] + 0.05*row[5]/N160
                    N160cs = N160*c_fines
                


            # Shear stress reduction factor (depth in meters)
            if row[1] > 20:
                warnings.warn('CSR (or equivalent rd values) at depths greater than about 20 m should be based on site response studies (Idriss and Boulanger, 2004)')

            if self.rd_method=='Idriss1999':
                # Idriss (1999), default value
                if row[1] <= 34:
                    rd = np.exp((-1.012-1.126*np.sin(row[1]/11.73+5.133)) + (0.106+0.118*np.sin(row[1]/11.28+5.142))*M)
                else:
                    rd = 0.12*exp(0.22*M)

            elif self.rd_method == 'LiaoWhitman1986':
                # Liao and Whitman (1986)
                if row[1] <= 9.15:
                    rd = 1 - 0.00765*row[1]
                else:
                    rd = 1.174 - 0.0267*row[1]
            elif self.rd_method == 'Golesorkhi1989':
                if row[1] <= 24:
                    rd = np.exp((-1.012-1.126*np.sin(row[1]/38.5+5.133)) + (0.106+0.118*np.sin(row[1]/37+5.142))*M)


            # Earthquake-induced cyclic stress ratio (CSR)                
            CSR = 0.65*sigmav/sigmavp*Pa*rd

            if row[4] == 1 or row[1] < Zw:
                CRR0 = 'n.a.'
                CRR = 'n.a.'
                FoS = 'n.a.'
                MSF = 'n.a'
                k_sigma = 'n.a.'
            else:
                # Magnitude scaling factor
                # Idriss (1999), default value
                MSF = min(1.8, 6.9 * np.exp(-M / 4) - 0.058)

                # Overburden correction factor
                # Boulanger and Idriss (2004)
                k_sigma = min(1.1, 1 - (min(0.3, 1 / (18.9 - 2.55 * np.sqrt(N160cs)))) * np.log(sigmavp / 100))

                # SPT Triggering correlation of liquefaction in clean sands
                # Idriss and Boulanger (2004)
                if N160cs < 37.5:
                    CRR0 = np.exp(N160cs / 14.1 + (N160cs / 126) ** 2 - (N160cs / 23.6) ** 3 + (N160cs / 25.4) ** 4 - 2.8)
                else:
                    CRR0 = 2

                # Cyclic resistance ratio (CRR)
                CRR = min(2., CRR0*MSF*k_sigma)
                if CRR/CSR > 2:
                    FoS = 2
                else:
                    FoS = CRR/CSR


            depth = row[1]
            gamma = row[6]

            output.append([row[1], ce, cb, cr, cs, N60, sigmav, sigmavp, CN, N160, delta_n, N160cs, rd, CSR, MSF, k_sigma, CRR0, CRR, FoS])

        self.new_bore_log_data = pd.DataFrame(output, columns=['depth', 'ce', 'cb', 'cr', 'cs', 'N60', 'sigmav', 'sigmavp', 'CN', 'N160', 'delta_n', 'N160cs', 'rd', 'CSR', 'MSF', 'k_simga', 'CRR0', 'CRR', 'FS'])


    # visualization of the liquefaction analysis
    def visualize(self):
        # subplot of SPT blow counts
        fig, ax = plt.subplots(ncols=3, figsize=(12, 6))
        [ax[x].xaxis.tick_top() for x in range(ax.shape[0])]
        [ax[x].xaxis.set_label_position('top') for x in range(ax.shape[0])]

        total_depth = max(self.new_bore_log_data['depth'])*-1.1

        soil_type_0 = ''
        depth_0 = 0
        spt_plot_max_x = max(self.new_bore_log_data.loc[self.new_bore_log_data.loc[:, 'N160'] != 'n.a.', 'N160'])*1.05
        for i, row in self.bore_log_data.iterrows():
            soil_type_1 = row[3]
            depth_1 = -row[1]
            ax[0].text(spt_plot_max_x*.05, depth_1, soil_type_1, color=(0, 0, 0, 0.4), verticalalignment='center')

            if not soil_type_0 == soil_type_1:
                ax[0].plot([0, spt_plot_max_x], [(depth_1+depth_0)*.5, (depth_1+depth_0)*.5], color=(0, 0, 0, 0.15))

            ax[0].plot([0, spt_plot_max_x], [depth_1, depth_1], **self.viz_dashed_guidelines)

            depth_0 = depth_1
            soil_type_0 = soil_type_1

        ax[0].scatter(self.bore_log_data.iloc[:, 2], -self.bore_log_data.iloc[:, 1], marker='x', label='$N_{SPT}$')
        ax[0].scatter(self.new_bore_log_data.loc[self.new_bore_log_data.loc[:, 'N160'] != 'n.a.', 'N160'],
                      -self.bore_log_data.ix[self.new_bore_log_data.loc[:, 'N160'] != 'n.a.', 1], marker='+', s=75, label='$(N_1)_{60}$')
        ax[0].legend(loc='lower right')
        ax[0].set(xlabel='SPT BLOW COUNT', ylabel='DEPTH ({})'.format(self.units_length), xlim=[0, spt_plot_max_x])
        ax[0].set_ylim(top=0, bottom=total_depth)

        # subplot of CSR & CRR
        depth_0 = 0
        layer_change_0 = 0
        liquefiable_0 = False
        csr_0 = 0
        crr_0 = 0
        na_0 = False

        csrcrr_plot_max_x = 1
        fos_plot_max_x = self.new_bore_log_data.loc[self.new_bore_log_data.loc[:, 'FS'] != 'n.a.', 'FS'].max()*1.1

        for i, row in self.new_bore_log_data.iterrows():
            depth_1 = -row[0]
            ax[1].plot([0, csrcrr_plot_max_x], [depth_1, depth_1], **self.viz_dashed_guidelines)
            ax[2].plot([0, fos_plot_max_x], [depth_1, depth_1], **self.viz_dashed_guidelines)
            na_1 = False
            csr_1 = row['CSR']
            crr_1 = row['CRR']
            if row['FS'] == 'n.a.':
                na_1 = True
                liquefiable_1 = False
            elif row['FS'] > 1:
                liquefiable_1 = False
            else:
                liquefiable_1 = True

            if i > 0:
                if not na_1 and not na_0:
                    ax[1].plot([csr_0, csr_1], [depth_0, depth_1], 'k--')
                    ax[1].plot([crr_0, crr_1], [depth_0, depth_1], 'k-')

            if not liquefiable_0 == liquefiable_1:
                layer_change_1 = (depth_1+depth_0)*.5
                if not liquefiable_1:
                    ax[2].text(0.5*fos_plot_max_x, (layer_change_0+layer_change_1)*0.5, 'LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)
                else:
                    ax[2].text(0.5*fos_plot_max_x, (layer_change_0 + layer_change_1) * 0.5, 'NON-LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)

                ax[1].plot([-1, csrcrr_plot_max_x], [(depth_1+depth_0)*.5, (depth_1+depth_0)*.5], color=(0, 0, 0, 0.15))
                ax[2].plot([0, fos_plot_max_x], [(depth_1 + depth_0) * .5, (depth_1 + depth_0) * .5], color=(0, 0, 0, 0.15))
                layer_change_0 = layer_change_1

            liquefiable_0 = liquefiable_1
            depth_0 = depth_1
            csr_0 = csr_1
            crr_0 = crr_1
            na_0 = na_1

        if liquefiable_1:
            ax[2].text(0.5*fos_plot_max_x, (total_depth+layer_change_1)*0.5, 'LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)
        else:
            ax[2].text(0.5 * fos_plot_max_x, (total_depth + layer_change_1) * 0.5, 'NON-LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)

        ax[1].plot([0, 0], [0, 0], 'k--', label='CSR')
        ax[1].plot([0, 0], [0, 0], 'k-', label='Earthquake-induced CRR')
        ax[1].legend(loc='lower right')
        ax[1].set(xlabel='CSR & CRR', xlim=[0, csrcrr_plot_max_x])
        ax[1].set_ylim(top=0, bottom=total_depth)

        # subplot of Factor of safety
        depth_0 = 0
        fs_0 = 0
        for i, row in self.new_bore_log_data.iterrows():
            fs_1 = row['FS']
            depth_1 = -row['depth']
            if i > 0 and fs_1 != 'n.a.' and fs_0 != 'n.a.':
                ax[2].plot([fs_0, fs_1], [depth_0, depth_1], 'k-')

            fs_0 = fs_1
            depth_0 = depth_1
        ax[2].plot([1, 1], [0, total_depth], '--', color=(0, 0, 0, 0.1))
        ax[2].set(xlabel='FACTOR OF SAFETY', xlim=[0, fos_plot_max_x])
        ax[2].set_ylim(top=0, bottom=total_depth)

        if self.name != None:
            fig.suptitle(self.name, fontsize=14, y=.99)
        plt.show()


    def save_to_file(self, file_name):
        self.new_bore_log_data.to_excel(file_name + '.xls')
        print(file_name + '.xls has been saved.')


    # Analytical methods for lateral spread and settlement analysis ************************************************************
    # Zhang, G., Robertson, P. K., & Brachman, R. W. I. (2004). Estimating liquefaction-induced lateral displacements using the standard penetration test or cone penetration test. Journal of Geotechnical and Geoenvironmental Engineering, 130(8), 861-871.
    def calc_ls_zhang2004(self, save_to_file=False, file_name='lateral_spread_analysis'):
        try:
            for i, row in self.new_bore_log_data.iterrows():
                if i == 0:
                    self.new_bore_log_data.loc[i, 'dHi'] = self.new_bore_log_data.loc[i, 'depth']
                elif i < len(self.new_bore_log_data) - 1:
                    self.new_bore_log_data.loc[i, 'dHi'] = (self.new_bore_log_data.loc[i + 1, 'depth'] - self.new_bore_log_data.loc[i - 1, 'depth']) / 2
                else:
                    self.new_bore_log_data.loc[i, 'dHi'] = self.new_bore_log_data.loc[i, 'depth'] - self.new_bore_log_data.loc[i - 1, 'depth']
                if self.new_bore_log_data.loc[i, 'FS'] == 'n.a.':
                    self.new_bore_log_data.loc[i, 'gamma_lim'] = 0
                    self.new_bore_log_data.loc[i, 'f_alpha'] = 0
                    self.new_bore_log_data.loc[i, 'gamma_max'] = 0
                    self.new_bore_log_data.loc[i, 'de'] = 0
                    self.new_bore_log_data.loc[i, 'de'] = 0
                else:
                    self.new_bore_log_data.loc[i, 'gamma_lim'] = max(0, min(0.5, 1.859*(1.1 - np.sqrt(self.new_bore_log_data.loc[i, 'N160cs'] / 45)) ** 3))
                    self.new_bore_log_data.loc[i, 'f_alpha'] = 0.032 + 0.69 * np.sqrt(max(7, self.new_bore_log_data.loc[i, 'N160cs'])) - 0.13 * max(7, self.new_bore_log_data.loc[i, 'N160cs'])
                    if row['FS'] > 2:
                        self.new_bore_log_data.loc[i, 'gamma_max'] = 0
                    elif row['FS'] < self.new_bore_log_data.loc[i, 'f_alpha']:
                        self.new_bore_log_data.loc[i, 'gamma_max'] = self.new_bore_log_data.loc[i, 'gamma_lim']
                    else:
                        self.new_bore_log_data.loc[i, 'gamma_max'] = min(self.new_bore_log_data.loc[i, 'gamma_lim'],
                                                            0.035 * (1 - self.new_bore_log_data.loc[i, 'f_alpha']) * (2 - row['FS']) / (
                                                            row['FS'] - self.new_bore_log_data.loc[i, 'f_alpha']))
                    self.new_bore_log_data.loc[i, 'de'] = 1.5 * np.exp(-0.369 * np.sqrt(self.new_bore_log_data.loc[i, 'N160cs'])) * min(0.08,
                                                                                                              self.new_bore_log_data.loc[
                                                                                                                  i, 'gamma_max'])
                self.new_bore_log_data.loc[i, 'dLDIi'] = self.new_bore_log_data.loc[i, 'dHi'] * self.new_bore_log_data.loc[i, 'gamma_max']
                self.new_bore_log_data.loc[i, 'dSi'] = self.new_bore_log_data.loc[i, 'dHi'] * self.new_bore_log_data.loc[i, 'de']
            print('LDI = {}, settlement = {}'.format(sum(self.new_bore_log_data.dLDIi.values), sum(self.new_bore_log_data.dSi.values)))

            if save_to_file:
                self.new_bore_log_data.to_excel(file_name + '.xls')
                print(file_name + '.xls has been saved.')

        except AttributeError:
            warnings.warn('Lateral spread and settlement analysis could not be done! Simplified liquefaction triggering analysis needs to be done first.')


if __name__ == '__main__':

    # ***********************************************************************************
    # Example on how to use this Python module with SPT-based liquefaction triggering analysis for a single boring (Soil Liquefaction During Earthquake textbook by Idriss and Boulanger)
    # ***********************************************************************************
    # 1. load the borehole data as a Panda's dataframe
    spt_idriss_boulanger_bore_data_appendix_a = pd.read_excel('default_datasets/spt_Idriss_Boulanger.xlsx')

    # 2. create a borehole object given the bore log data from Appendix A of Idriss and Boulanger textbook
    log1 = Borehole(spt_idriss_boulanger_bore_data_appendix_a)

    # 3. run simplified liquefaction triggering method on log1 to calculate factors of safety
    log1.simplified_liquefaction_triggering_fos(Pa=0.280, M=6.9, Zw=1.8, sampler_correction_factor=1, liner_correction_factor=1, hammer_energy=75, rod_extension=1.5)

    # 4. (optional) visualize the output
    log1.visualize()

    # 5. (optional) save the output to an EXCEL file
    log1.save_to_file('triggering_analysis_on_log_from_Idriss_and_Boulanger')

    # 6. (optional) run lateral spread and settlement analysis based on Zhang & Robinson's model
    # NOTE: you need to run simplified_liquefaction_triggering_fos() Method on the log before running this Method
    log1.calc_ls_zhang2004(save_to_file=True, file_name='zhang2004_lateral_spread_analysis')




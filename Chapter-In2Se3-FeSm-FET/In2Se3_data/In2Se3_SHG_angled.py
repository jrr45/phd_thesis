# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:21:28 2019

@author: Justin
"""

import sys
import os
sys.path.append(os.path.join('..', '..', 'Code'))
import material_plotter as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.projections.polar import ThetaLocator

working_dir = r'SHG_Bill'

SHG_csv_files = ['In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_1_1.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_1_2.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_1_3.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_1_3_2.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_1_4.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_2_1.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_2_2.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_2_3.txt',
                 'In2Se3_SiO2Si_03262019_pol_5mW_50xLWD_55NA_1000ms_100umslit_2_3_1.txt',
                 ]

def r_pl_file(file):
    print(file)
    if not os.path.isfile(file):
        raise Exception("not a file %s", file)
        return []

    """plot data from file"""
    data = np.genfromtxt(file, names=['Angle_Rad','avgIntensityNorm', 'avgIntensityPowerCorrectedNorm'], 
                         dtype='<f8', delimiter=',')
    
    return data

def plot_angled_SHG_general(filename, savename):

    plt.rcParams["font.family"] = "Arial"
    
    fig = plt.figure(figsize=(2, 2), dpi=300)
    ax = fig.add_subplot(111, projection='polar')
    
    ax.tick_params(axis='x', which='major', labelsize=10, pad=-4)
    
    ax.set_rticks([.5,1])
    ax.set_yticklabels(['']*2)
    
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 6, endpoint=False))
    ax.spines['polar'].set_visible(False)
    
    data = r_pl_file(os.path.join(working_dir, filename))
    
    ax.plot(data['Angle_Rad'], data['avgIntensityPowerCorrectedNorm'], '.-', ms=3, linewidth=1, color=mp.colors_set1[1])
    
    
    mp.save_generic_svg(fig, working_dir, savename)
    plt.show()
    plt.clf()

def plot_angled_SHG():
    for filename in SHG_csv_files[:]:
        plot_angled_SHG_general(filename, filename.replace('.txt', ''))

def main():
    plot_angled_SHG()

if __name__== "__main__":
  main()
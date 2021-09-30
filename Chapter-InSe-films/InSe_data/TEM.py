# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
sys.path.append(os.path.join('..', '..', 'Code'))
import material_plotter as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import hyperspy as hs

TEM_film = mp.device()
TEM_film.fileroot = 'TEM_data'

TEM_scan_files = [
    '09.50.01 CCD Acquire_0006.txt',
    '09.50.01 CCD Acquire_0013.txt',
    '09.50.01 CCD Acquire_0016.txt',
    ]

def plot_TEM_maps():
    for file in TEM_scan_files:
        element_file = os.path.join('TEM_data', file)
    
        eds2_data = np.genfromtxt(element_file, delimiter=",")
        
        cmap1 = plt.get_cmap('gray')
    
        #eds2_se = np.nan_to_num(eds2_se)
        fig = plt.figure(figsize=(2, 2), dpi=600)
        ax = fig.add_subplot(111)
        ax.imshow(eds2_data, cmap=cmap1)
        plt.axis('off')
        #plt.colorbar()
        plt.show()
        mp.save_generic_svg(fig, TEM_film,  '_' + os.path.splitext(file)[0])
    
def main():
    plot_TEM_maps()

if __name__== "__main__":
  main()

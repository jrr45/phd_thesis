# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:31:37 2019

@author: Justin
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Jpython_plotter as jpp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

inse_dir = r'Raman-PL_Kazu'
inse_pl_filenames = ['F1-P1-60s_2times_10per160uW_x100_pl.txt',
                     'F1-P2-60s_2times_10per160uW_x100_pl.txt',
                     'F2-P1-60s_2times_10per160uW_x100_pl.txt',
                     'F2-P2-60s_2times_10per160uW_x100_pl.txt',
                     'F3-P1-60s_2times_10per160uW_x100_pl.txt',
                     'F3-P2-60s_2times_10per160uW_x100_pl.txt',
                    ]
inse_rm_filenames = ['F1-P1-60s_2times_10per160uW_x100_r.txt',
                     'F1-P2-60s_2times_10per160uW_x100_r.txt',
                     'F2-P1-60s_2times_10per160uW_x100_r.txt',
                     'F2-P2-60s_2times_10per160uW_x100_r.txt',
                     'F3-P1-60s_2times_10per160uW_x100_r.txt',
                     'F3-P2-60s_2times_10per160uW_x100_r.txt',
                    ]
def r_pl_file(file):
    print(file)
    if not os.path.isfile(file):
        raise Exception("not a file %s", file)
        return []

    """plot data from file"""
    data = np.genfromtxt(file, names=['Energy','Intensity'], 
                         dtype='<f8', delimiter='\t')
    
    return data
    
    
def plot_raman_general(filename):
    file = r_pl_file(os.path.join(inse_dir, filename))
    fig = plt.figure(figsize=(1.9, 1.9), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["Raman Shift ($cm^{-1}$)", 'Intensity (Arb. Unit) '],
                             yscale='linear', fontsize=10, labelsize=10, labelpad=[0,3]) #fontsize=10, labelsize=10
    ax.tick_params(axis='both', which='major', pad=1)
    
    ax.plot(file['Energy'], file['Intensity'], '.-', ms=3, linewidth=1.5, color=jpp.colors_set1[1])
    ax.set_yticklabels(['']*4)
    
    inc = max(file['Intensity'])/4
    ax.yaxis.set_major_locator(MultipleLocator(inc))
    
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    ax.set_xlim((90, 320))
    
    jpp.save_generic_svg(fig, inse_dir, filename.replace('.txt', ''))
    plt.show()
    plt.clf()

def plot_pl_general(filename):
    file = r_pl_file(os.path.join(inse_dir, filename))
    fig = plt.figure(figsize=(1.9, 1.9), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["Energy (eV)", 'Intensity (Arb. Unit) '],
                             yscale='linear', fontsize=10, labelsize=10, labelpad=[0,3]) #fontsize=10, labelsize=10
    inc = 10
    ax.plot(file['Energy'][0::inc], file['Intensity'][0::inc], '.-', ms=3, linewidth=1.5, color=jpp.colors_set1[1])
    ax.set_yticklabels(['']*4)
    
    inc = max(file['Intensity'])/4
    ax.yaxis.set_major_locator(MultipleLocator(inc))
    
    jpp.save_generic_svg(fig, inse_dir, filename.replace('.txt', ''))
    plt.show()
    plt.clf()

def plot_raman():
    for file in inse_rm_filenames:
        plot_raman_general(file)

def plot_PL():
    for file in inse_pl_filenames:
        plot_pl_general(file)


def main():
    plot_raman()
    plot_PL()

if __name__== "__main__":
  main()

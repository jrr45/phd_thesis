# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:31:37 2019

@author: Justin
"""

import sys
import os
sys.path.append(os.path.join('..', '..', 'Code'))
import material_plotter as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


inse_dir = r'films-Maria'
film_device = mp.device()
film_device.fileroot = os.path.join(inse_dir)

inse_pl_filenames = ['03-center_PL_MBE2-200307A-MH_19_26_58_2020_05_13_488 nm ULF_600 (500nm)_25%_01.txt',
                     '03-center_PL_MBE2-200307B-MH_19_57_10_2020_05_13_488 nm ULF_600 (500nm)_25%_01.txt',
                    ]
inse_rm_filenames = ['01-center_Raman_MBE2-200307A-MH_18_59_04_2020_05_13_488 nm ULF_1800 (500nm)_25%_01.txt',
                     '01-center_Raman_MBE2-200307B-MH_19_37_16_2020_05_13_488 nm ULF_1800 (500nm)_25%_01.txt',
                     '02-center_Raman_MBE2-200307A-MH_19_14_09_2020_05_13_488 nm ULF_1800 (500nm)_25%_01.txt',
                     '02-center_Raman_MBE2-200307B-MH_19_44_05_2020_05_13_488 nm ULF_1800 (500nm)_25%_01.txt',
                    ]
inse_xrd_filenames = ['MBE2-200307A-MH_2theta-omega_1.csv',
                      'MBE2-200307B-MH_2theta-omega_1.csv',
                     ]


def r_pl_file(file):
    print(file)
    if not os.path.isfile(file):
        raise Exception("not a file %s", file)
        return []

    """plot data from file"""
    data = np.genfromtxt(file, names=['Energy','Intensity'], 
                         dtype='<f8', delimiter='\t', comments='#')
    
    return data

def r_xrd_file(file):
    print(file)
    if not os.path.isfile(file):
        raise Exception("not a file %s", file)
        return []

    """plot data from file"""
    data = np.genfromtxt(file, names=['Angle','Intensity'], 
                         dtype='<f8', delimiter=',', skip_header=34)
    
    return data
    
    
def plot_raman_general(filename):
    file = r_pl_file(os.path.join(inse_dir, filename))
    fig = plt.figure(figsize=(1.9, 1.9), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["Raman Shift ($cm^{-1}$)", 'Intensity (Arb. Unit) '],
                             yscale='linear', fontsize=10, labelsize=10, labelpad=[0,3]) #fontsize=10, labelsize=10
    ax.tick_params(axis='both', which='major', pad=1)
    
    energy = file['Energy']
    intensity = file['Intensity']
    ind = np.where(energy > 5)
    energy = energy[ind]
    intensity = intensity[ind]
    
    
    ax.plot(energy, intensity, '.-', ms=0, linewidth=1., color=mp.colors_set1[1])
    ax.set_yticklabels(['']*4)
    
    inc = max(intensity)/4
    ax.yaxis.set_major_locator(MultipleLocator(inc))
    
    if np.max(energy) < 400:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim((50, 325))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim((100, 325))
        
    mp.save_generic_svg(fig, film_device, filename.replace('.txt', ''))
    plt.show()
    plt.clf()

def plot_pl_general(filename):
    file = r_pl_file(os.path.join(inse_dir, filename))
    fig = plt.figure(figsize=(1.9, 1.9), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["Energy (eV)", 'Intensity (Arb. Unit) '],
                             yscale='linear', fontsize=10, labelsize=10, labelpad=[0,3]) #fontsize=10, labelsize=10
    
    energy = file['Energy']
    intensity = file['Intensity']

    inc = 1
    energy = energy[0::inc]
    intensity = intensity[0::inc]
    
    ax.plot(energy, intensity, '.-', ms=1, linewidth=1.5, color=mp.colors_set1[1])
    ax.set_yticklabels(['']*4)
    
    inc = max(file['Intensity'])/4
    ax.yaxis.set_major_locator(MultipleLocator(inc))
    
    ax.set_xlim((1.3, 1.73))
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    
    mp.save_generic_svg(fig, film_device, filename.replace('.txt', ''))
    plt.show()
    plt.clf()
    
def plot_xrd_general(filename):
    file = r_xrd_file(os.path.join(inse_dir, filename))
    fig = plt.figure(figsize=(1.9, 1.9), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["2Θ (degrees)", 'Intensity (Arb. Unit) '],
                             yscale='log', fontsize=10, labelsize=10, labelpad=[0,3]) #fontsize=10, labelsize=10
    
    angle = file['Angle']
    intensity = file['Intensity']
    
    inc = 1
    angle = angle[0::inc]
    intensity = intensity[0::inc]
    
    ax.plot(angle, intensity, '.-', ms=0, linewidth=0.5, color=mp.colors_set1[1])
    ax.set(yticklabels=[])
    
    ax.set_xlim((0, None))
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.axes.yaxis.set_ticklabels([])
    ax.set_ylim((2, None))
    
    mp.save_generic_svg(fig, film_device, filename.replace('.csv', ''))
    plt.show()
    plt.clf()

def plot_raman():
    for file in inse_rm_filenames:
        plot_raman_general(file)

def plot_PL():
    for file in inse_pl_filenames:
        plot_pl_general(file)
        
def plot_XRD():
    for file in inse_xrd_filenames:
        plot_xrd_general(file)


def main():
    plot_raman()
    plot_PL()
    plot_XRD()

if __name__== "__main__":
  main()

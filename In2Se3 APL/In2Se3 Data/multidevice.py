# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:19:16 2021

@author: ffdra
"""

import os
import sys
sys.path.append(os.path.join('..','..', 'Code'))
import material_plotter as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator)

import JR200115_11 as sampleA
import JR200115_17 as sampleB
import JR190919_03 as sampleC
import JR190815_04 as sampleD


def get_width_VT_vs_T(sample, Npoints=5, Icutoff=10*10**-11):
    filenames = sample.RTloop_filenames
    files = [mp.process_file(os.path.join(sample.fileroot, x)) for x in filenames]
    
    ΔVT = []
    T = []
    for file in files:
        T.append(file['Temperature_K'][0])
        VTi, _, _, VTd, _, _ = mp.calc_max_IVG_slope(None, file, Npoints=Npoints,
                                                          subplot=False, Icutoff=Icutoff)
        ΔVT.append(VTd-VTi)
    
    return T, ΔVT

def plot_width_VT(savename, log=False, size=2, Npoints=5, Icutoff=1*10**-11):
    T_A, VG_A = get_width_VT_vs_T(sampleA, Npoints=Npoints, Icutoff=Icutoff)
    T_B, VG_B = get_width_VT_vs_T(sampleB, Npoints=Npoints, Icutoff=Icutoff)
    T_C, VG_C = get_width_VT_vs_T(sampleC, Npoints=Npoints, Icutoff=Icutoff)
    T_D, VG_D = get_width_VT_vs_T(sampleD, Npoints=Npoints, Icutoff=Icutoff)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{ΔV_{T}}$ (V)'],
                             yscale=('log' if log else 'linear'), fontsize=10, labelsize=10)
    
    ax.plot(T_A, VG_A, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[0])
    ax.plot(T_B, VG_B, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[1])
    ax.plot(T_C, VG_C, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[2])
    ax.plot(T_D, VG_D, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[3])
    
    ax.set_ylim(0, None)
    ax.set_xlim(0, 322)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    mp.save_generic_svg(fig, '.', savename)

def get_width_Vg_vs_T(sample, current):
    filenames = sample.RTloop_filenames
    files = [mp.process_file(os.path.join(sample.fileroot, x)) for x in filenames]
    
    DVG = []
    T = []
    for file in files:
        T.append(file['Temperature_K'][0])
        dVg = mp.width_Vg(file, current)
        DVG.append(dVg)
    
    return T, DVG

def plot_width_Vg(current, savename, log=False, size=2):
    T_A, VG_A = get_width_Vg_vs_T(sampleA, current)
    T_B, VG_B = get_width_Vg_vs_T(sampleB, current)
    T_C, VG_C = get_width_Vg_vs_T(sampleC, current)
    T_D, VG_D = get_width_Vg_vs_T(sampleD, current)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{ΔV_{G}}$ (V)'],
                             yscale=('log' if log else 'linear'), fontsize=10, labelsize=10)
    
    ax.plot(T_A, VG_A, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[0])
    ax.plot(T_B, VG_B, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[1])
    ax.plot(T_C, VG_C, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[2])
    ax.plot(T_D, VG_D, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[3])
    
    ax.set_ylim(0, None)
    ax.set_xlim(0, 322)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    mp.save_generic_svg(fig, '.', savename)

def main():
    plot_width_Vg(current=1*10.**-10, savename="_100pA_")
    plot_width_Vg(current=1*10.**-9, savename="_1nA_")
    plot_width_Vg(current=1*10.**-8, savename="_10nA_")
    plot_width_Vg(current=1*10.**-7, savename="_100nA_")
    plot_width_Vg(current=1*10.**-6, savename="_1uA_")
    plot_width_Vg(current=1*10.**-5, savename="_10uA_")
    plot_width_VT("_ΔVT_vs_T", log=False, size=2, Npoints=5, Icutoff=1*10**-11)

if __name__== "__main__":
  main()
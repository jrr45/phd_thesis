# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:23:07 2019

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

fileroot = r'Bulk'

def plot_rho(figsize=(2, 2)):
    colors = jpp.colors_set1
    filenames = ['Bulk_rect_010_RvsT_hall.txt',
                 'Bulk_rect_002_RvsT_c-plane.txt']
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\\rho}$ (kΩ$\cdot$cm)'],
                             yscale='linear', size=1.9)

    temperaturesab = files[0]['Temperature_K']
    Rxxab = files[0]['Resistance_1_Ohms']
    rhoxxab = Rxxab*0.34*0.05/0.11
    temperaturesc = files[0]['Temperature_K']
    Rxxc = files[1]['Resistance_1_Ohms']
    rhoxxc = Rxxc*0.34*0.11/0.05
    ax.plot(temperaturesab, rhoxxab*.001, '.-', ms=3, linewidth=1.5, color=colors[0])
    ax.plot(temperaturesc, rhoxxc*.001, '.-', ms=3, linewidth=1.5, color=colors[1])
    
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    jpp.save_generic_svg(fig, fileroot, "_Bulk_RT")

def plot_rho_ln(figsize=(2, 2)):
    colors = jpp.colors_set1
    filenames = ['Bulk_rect_010_RvsT_hall.txt',
                 'Bulk_rect_002_RvsT_c-plane.txt']
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T^{-1/4}}$ ($K^{-1/4}$)", '$\it{\\rho}$ ($Ω\cdot$cm)'],
                             yscale='log', size=1.9, fontsize=10, labelsize=10)

    temperaturesab = files[0]['Temperature_K']**(-1.0/4.0)
    Rxxab = files[0]['Resistance_1_Ohms']
    rhoxxab = Rxxab*0.34*0.05/0.11
    temperaturesc = files[1]['Temperature_K']**(-1.0/4.0)
    Rxxc = files[0]['Resistance_1_Ohms']
    rhoxxc = Rxxc*0.34*0.11/0.05
    ax.plot(temperaturesab, rhoxxab*.001, '.-', ms=3, linewidth=1.5, color=colors[0])
    ax.plot(temperaturesc, rhoxxc*.001, '.-', ms=3, linewidth=1.5, color=colors[1])
    
    #ax.xaxis.set_major_locator(MultipleLocator(.25))
    #ax2.xaxis.set_major_locator(MultipleLocator(.25))
    ax.set_ylim((10**-2,None))
    
    jpp.save_generic_svg(fig, fileroot, "_Bulk_RT_ln_14")

def main():
    plot_rho(figsize=(2, 2))
    plot_rho_ln(figsize=(2, 2))

if __name__== "__main__":
  main()
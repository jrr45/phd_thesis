# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:33:56 2019

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join('..','..', 'Code'))
import material_plotter as mp

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from scipy import stats
from matplotlib.ticker import (MultipleLocator)

fileroot = "JR190919_03"
JR190919_03_width = 10*10**-6 # meters
JR190919_03_length = 20*10**-6 # meters
JR190919_03_volt_spacing = 8*10**-6 # meters (center to center)


RTloop_filenames = [
        'JR190919_03_005_RvsVg_300.0K_loops_v2.txt',
    'JR190919_03_006_RvsVg_270.0K_loops.txt',
    'JR190919_03_007_RvsVg_250.0K_loops.txt',
    'JR190919_03_008_RvsVg_200.0K_loops.txt',
    'JR190919_03_009_RvsVg_150.0K_loops.txt',
    'JR190919_03_010_RvsVg_100.0K_loops.txt',
    'JR190919_03_011_RvsVg_90.0K_loops.txt',
    'JR190919_03_012_RvsVg_80.0K_loops.txt',
    'JR190919_03_013_RvsVg_70.0K_loops.txt',
    'JR190919_03_014_RvsVg_60.0K_loops.txt',
    'JR190919_03_015_RvsVg_50.0K_loops.txt',
    'JR190919_03_016_RvsVg_40.0K_loops.txt',
    'JR190919_03_017_RvsVg_30.0K_loops.txt',
    'JR190919_03_018_RvsVg_20.0K_loops.txt',
    'JR190919_03_019_RvsVg_10.0K_loops.txt',
    'JR190919_03_020_RvsVg_4.0K_loops.txt'
    ]

def plot_delta_Vgmax():
    colors = ['#0000FF', '#FF0000', '#000000']
              
    filenames = RTloop_filenames
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=(3, 3*.90), dpi=300)
    
    dgv = []
    Imax = []
    temp = []
    for file in files:
        x = mp.max_Vg(file)
        dgv.append(x[0])
        Imax.append(x[2]*(10**9))
        temp.append(file['Temperature_K'][0])
    
    
    (ax, ax2, ax3) = mp.pretty_plot(fig, labels=["$\it{T}$ (K)", '$\it{ΔV_{G}^{max}}$ (V)', '$\it{@I_{D}}$ (nA)'],
                             yscale=['linear','linear'])
    
    ax.plot(temp, dgv, '.-', ms=3, linewidth=1.5, color=colors[0])
    ax2.plot(temp, Imax, '.-', ms=3, linewidth=1.5, color=colors[1])
    
    #ax.set_xlim([0, 300])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    plt.tight_layout()
    
    mp.save_generic_svg(fig, fileroot, "_Vg_max")
    
def plot_IDSvsVg_effect_heating(figsize=2, log=False):
    colors = [mp.colors_set1[1], mp.colors_set1[0]]
    filenames = ['JR190919_03_002_RvsVg_300.0K_initial.txt',
                 'JR190919_03_005_RvsVg_300.0K_loops.txt']
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    files = mp.slice_data_each(files, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
    
    mp.plot_IDvsVg_generic(fileroot, files, '_JR190919_03_effect_heating', colors, log=log, size=figsize, \
                             majorx=40, ylim=(None,None))
    
def plot_IDSvsVDS_effect_heating(figsize=2):
    colors = [mp.colors_set1[1], mp.colors_set1[0]]
    filenames = ['JR190919_03_001_IvsV_initial.txt',
                 'JR190919_03_004_IvsV_300.0K_after_heat.txt']
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    mp.plot_IDvsVDS_generic(fileroot, files, '_JR190919_03_effect_heating', colors,  \
                              log=False, invertaxes=False, size=figsize, xadj=0, x_mult=.1)

    
def remove_leak_by_fit(file, start=-75., end=-75., mid=75., flat1=-40., flat2=40.):
    #returns V_G, I_DS-I_Leakfit, I_leakfit
    occ0 = mp.first_occurance_1D(file['Gate_Voltage_V'], start, tol=0.2, starting_index=0)
    occ1 = mp.first_occurance_1D(file['Gate_Voltage_V'], end, tol=0.2, starting_index=occ0+1)
    
    
    V_G = file['Gate_Voltage_V'][occ0:occ0+occ1+2]
    I_DS = file['Current_A'][occ0:occ0+occ1+2]
    I_Leak = file['Gate_Leak_Current_A'][occ0:occ0+occ1+2]
        
    occ_flat0 = mp.first_occurance_1D(V_G, flat1, tol=0.2, starting_index=0)
    occ_flat1 = mp.first_occurance_1D(V_G, flat2, tol=0.2, starting_index=0)

    slope1, intercept1, r_value, p_value, std_err = stats.linregress(V_G[occ_flat0:occ_flat1], I_Leak[occ_flat0:occ_flat1])
    
    def fitline1(x, a,b,c,d):
        return a + b*(np.exp(-c*x + d))
    
    initial = None#[10**-12,10**-12,10**-3, -60, 10**-12]
    def fitlinesplit1(x, p):
        return fitline1(x, p[0],p[1],p[2],p[3])
    
    occ_mid = mp.first_occurance_1D(V_G, mid, tol=0.2, starting_index=0)

    #fit madness
    GVvals1 = V_G[0:occ_mid+1]
    Ivals1 = I_Leak[0:occ_mid+1] - GVvals1*slope1 - intercept1

    popt1, pcov1 = optimize.curve_fit(fitline1, GVvals1, Ivals1, p0=initial, maxfev=1000000)
    
    I_leak_fit1 = fitlinesplit1(GVvals1, popt1) + GVvals1*slope1 + intercept1
    
    #ax.plot(file['Gate_Voltage_V'], (fitlinesplit1(GVvals, popt) + GVvals*slope)*I_pow, '-', ms=3, linewidth=.5, color='black')
    #print(popt)
        
    def fitline2(x, a,b,c,d):
        return a + b*(np.exp(c*x + d))
    def fitlinesplit2(x, p):
        return fitline2(x, p[0],p[1],p[2],p[3])

    occ_flat3 = mp.first_occurance_1D(V_G, flat2, tol=0.2, starting_index=occ_mid) + occ_mid
    occ_flat4 = mp.first_occurance_1D(V_G, flat1, tol=0.2, starting_index=occ_mid) + occ_mid
    
    slope2, intercept2, r_value, p_value, std_err = stats.linregress(V_G[occ_flat3:occ_flat4], I_Leak[occ_flat3:occ_flat4])
    
    GVvals2 = V_G[occ_mid:]
    Ivals2 = I_Leak[occ_mid:] - GVvals2*slope2 - intercept2

    popt2, pcov2 = optimize.curve_fit(fitline2, GVvals2, Ivals2, p0=initial, maxfev=1000000)
    #popt, pcov = optimize.curve_fit(fitline2, GVvals, Ivals, p0=initial, maxfev=1000000, ftol=1.0e-15, xtol=1.0e-15)#, ftol=1.0e-18, xtol=1.0e-15)
    #ax.plot(file['Gate_Voltage_V'], (fitlinesplit2(GVvals, popt) + GVvals*slope)*I_pow, '-', ms=3, linewidth=.5, color='black')
    
    I_leak_fit2 = GVvals2*slope2 + intercept2 + fitlinesplit2(GVvals2, popt2)

    return (V_G, I_DS, I_Leak, I_leak_fit1, I_leak_fit2)
       
def plot_loopI():
    colors = mp.colors_set1
    colors = [colors[0], colors[3], colors[2], colors[1]]
              
    filenames = RTloop_filenames
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    index0V = [mp.first_occurance_1D(file['Gate_Voltage_V'], 0) for file in files]
    index25V = [mp.first_occurance_1D(file['Gate_Voltage_V'], 25) for file in files]
    index50V = [mp.first_occurance_1D(file['Gate_Voltage_V'], 50) for file in files]
    index75V = [mp.first_occurance_1D(file['Gate_Voltage_V'], 75) for file in files]
    
    fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{I_{D}}$ (μA)'],
                             yscale='linear', fontsize=10)
        
    I0V = [(file['Current_A'][i])*(10**6) for (file, i) in zip(files, index0V)]
    I25V = [(file['Current_A'][i])*(10**6) for (file, i) in zip(files, index25V)]
    I50V = [(file['Current_A'][i])*(10**6) for (file, i) in zip(files, index50V)]
    I75V = [(file['Current_A'][i])*(10**6) for (file, i) in zip(files, index75V)]
    
    temper = [(file['Temperature_K'][0]) for file in files]
    ax.plot(temper, I0V, '.-', ms=3, linewidth=1.5, color=colors[0])
    ax.plot(temper, I25V, '.-', ms=3, linewidth=1.5, color=colors[1])
    ax.plot(temper, I50V, '.-', ms=3, linewidth=1.5, color=colors[2])
    ax.plot(temper, I75V, '.-', ms=3, linewidth=1.5, color=colors[3])
    
    ax.set_ylim((None, 3.6))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim((None, 320))
    
    #plt.tight_layout()
    mp.save_generic_svg(fig, fileroot, "_loop_I")       

def plot_300K_IDvsVDS(figsize=2, log=False):
    colors = mp.colors_set1[[0,4,3,6,2,8,1]]
    
    savename = '_JR190919_03_300K'
    
    start = 31
    filenames = [('JR190919_03_0' + str(i) + '_IvsV_300.0K_positive.txt') for i in range(start, start+7)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_positive_increasing', colors,\
                              size=figsize, log=log, xadj=0)
    
    filenames = [('JR190919_03_0' + str(i) + '_IvsV_300.0K_positive.txt') for i in range(start+7, start+14)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_positive_decreasing', colors[::-1],\
                              size=figsize, log=log, xadj=0)
    
    filenames = [('JR190919_03_0' + str(i) + '_IvsV_300.0K_negative.txt') for i in range(start+14, start+21)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_negative_increasing', colors,\
                              invertaxes=True, size=figsize, log=log, xadj=0)
    
    filenames = [('JR190919_03_0' + str(i) + '_IvsV_300.0K_negative.txt') for i in range(start+21, start+28)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_negative_decreasing', colors[::-1],\
                              invertaxes=True, size=figsize, log=log, xadj=0)
    

def plot_270K_IDvsVDS(figsize=2, log=False):
    colors = mp.colors_set1[[0,4,3,6,2,8,1]]
    
    savename = '_JR190919_03_270K'
    
    start = 61
    filenames = [('JR190919_03_' + str(i).zfill(3) + '_IvsV_270.0K_positive.txt') for i in range(start, start+31, 5)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_positive_increasing', colors,\
                              figsize=figsize, log=log, xadj=1)
    
    filenames = [('JR190919_03_' + str(i).zfill(3) + '_IvsV_270.0K_positive.txt') for i in range(start+31, start+62, 5)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_positive_decreasing', colors[::-1],\
                              figsize=figsize, log=log, xadj=1)
    
    filenames = [('JR190919_03_' + str(i).zfill(3) + '_IvsV_270.0K_negative.txt') for i in range(start+62, start+93,5)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_negative_increasing', colors,\
                              invertaxes=True, figsize=figsize, log=log, xadj=1)
    
    filenames = [('JR190919_03_' + str(i).zfill(3) + '_IvsV_270.0K_negative.txt') for i in range(start+93, start+124, 5)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + '_negative_decreasing', colors[::-1],\
                              invertaxes=True, figsize=figsize, log=log, xadj=1)

def plot_IDSvsT_cooling_vs_warming(size=2):
    colors = mp.colors_set1
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{I_{D}}$ (A)'],
                             yscale='log', fontsize=10, labelsize=10)
    
    filenames = ['JR190919_03_188_RvsT_75.00Vg_warm.txt',
                 'JR190919_03_186_RvsT_75.00Vg_cool.txt',
                 #'JR190919_03_193_RvsT_75.00Vg_warm.txt',
                 #'JR190919_03_022_RvsT_75.00Vg_warm.txt',
                 ]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    for file, i in zip(files, range(len(colors))):
        wheres = np.where(file['Current_A'] > 5.0*10**-11 )
        temps = file['Temperature_K'][wheres]
        Is = file['Current_A'][wheres]
        ax.plot(temps, Is, '.-', ms=3, linewidth=1.5, color=colors[i])
    
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    ax.set_xlim((None, 330))
    ax.set_ylim((None, 2.4*10**-6))
    
    mp.save_generic_svg(fig, fileroot, "_JR190919_03_IDS75V(T)_cooling_vs_warming")
    
def plot_RT4p(size=2):
    colors = mp.colors_set1
    colors = [colors[1], colors[0], colors[3]]#013
              
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{R}$ (Ω)'],
                             yscale='log', fontsize=10, labelsize=10)
    
    filenames = ['JR190919_03_188_RvsT_75.00Vg_warm.txt',
                 #'JR190919_03_186_RvsT_75.00Vg_cool.txt',
                 'JR190919_03_193_RvsT_75.00Vg_warm.txt',
                 'JR190919_03_022_RvsT_75.00Vg_warm.txt',
                 ]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    for file, color in zip(files, colors):
        ax.plot(file['Temperature_K'], abs(file['Resistance_2_Ohms']), '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(None, 315)
    
    mp.save_generic_svg(fig, fileroot, "_JR190919_03_R4pt(T)_coolwarm")
    
def plot_R_DSvT_4p(size=2):
    colors = mp.colors_set1
    colors = [colors[1], colors[0], colors[3]]#013
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{R_{DS}}$ (Ω)'],
                             yscale='log', fontsize=10, labelsize=10)
    
    filenames = ['JR190919_03_188_RvsT_75.00Vg_warm.txt',
                 #'JR190919_03_186_RvsT_75.00Vg_cool.txt',
                 'JR190919_03_193_RvsT_75.00Vg_warm.txt',
                 'JR190919_03_022_RvsT_75.00Vg_warm.txt',
                 ]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    for file, color in zip(files, colors):
        ax.plot(file['Temperature_K'], abs(file['Resistance_1_Ohms']), '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(None, 315)
    
    mp.save_generic_svg(fig, fileroot, "_JR190919_03_R_DSvT_4p_coolwarm")

def plot_IDSvsVg_Vs(): # unused, not much useful to say
    colors = mp.colors_set1
    filenames = ['JR190919_03_005_RvsVg_300.0K_loops.txt',
                 'JR190919_03_059_RvsVg_300.0K_loop.txt']
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.rate_300K_filenames(fileroot, files, '_JR190919_03_Vs', colors)

def calc_field_effect_mobility(fileroot, filenames, volt_range=(0,10)):
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    files = mp.slice_data_each(files, "Voltage_1_V", volt_range[0], volt_range[1],\
                                 max(volt_range[0], volt_range[1])/100)
    print(files)
    

def main(): #Sample C
    show_all = False
    figsize = 2
    
    # -- 300K ID vs VDS curves
    if False or show_all:
        plot_300K_IDvsVDS(figsize=figsize, log=False)
        plot_300K_IDvsVDS(figsize=figsize, log=True)
    
    # Plot ID vs VG loops
    if False or show_all:
        mp.plot_IDvsVg_each(fileroot, RTloop_filenames, '_JR190919_03', log=False, size=figsize, majorx=40,
                          ylim=(None,None), fontsize=10, labelsize=10)
        mp.plot_IDvsVg_each(fileroot, RTloop_filenames, '_JR190919_03', log=True, size=figsize, majorx=40,
                          ylim=(None,None), fontsize=10, labelsize=10)
        mp.plot_IDvsVg_each(fileroot, [RTloop_filenames[3]], '_JR190919_03', log=True, size=figsize, majorx=40,
                          ylim=(10**-12, 1*10**-6), fontsize=10, labelsize=10) # adjust for text
    
     # -- Cross section of loop data --
    if False or show_all:
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR190919_03_RDS", increments=[0,25,50,75],\
                                      figsize=figsize, log=True, xlim=(0,322), ylim=(None, None))
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR190919_03_RDSv2", increments=[50,75],\
                                      figsize=figsize, log=True, xlim=(0,322), ylim=(None, None), colororder=[2,1])
        
    # -- 270K ID vs VDS curves
    if False:
        plot_270K_IDvsVDS(figsize=figsize, log=False)
        plot_270K_IDvsVDS(figsize=figsize, log=True)
    
    # -- Effect off heating sample on ID
    if False:
        plot_IDSvsVg_effect_heating()
        plot_IDSvsVDS_effect_heating()
        
    # -- 2 and 4pt resistance, constant gating changing T
    if False:
        plot_RT4p(size=figsize)
        plot_R_DSvT_4p(size=figsize)
    
    # -- cooling vs warming difference
    if False:
        plot_IDSvsT_cooling_vs_warming(size=figsize)
    
    # unused but saving
    if False:    
        plot_delta_Vgmax()
        plot_loopI()

    # -- carrier mobility μ
    if False or show_all:
        mp.plot_mobility_μ_cross_section(fileroot, RTloop_filenames, "_JR190919_03", JR190919_03_length, JR190919_03_width, figsize=1.5, ylim=(None, None),\
                                           log=False, increments=[25, 50, 75], colororder=[3,2,1])
            
    #print("maximal drain current %s µAµm^-1" % (14.854492293E-9*1000)/)
    #calc_field_effect_mobility(fileroot, ['JR190919_03_037_IvsV_300.0K_positive.txt',], (0,1.5))
    #mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR190919_03_RDS", increments=[0,25,50,75],\
    #                                  figsize=1.5, log=True, ylim=(None, 10**9), fontsize=14, labelsize=14, xinc=150)
    
    # min subthreshold slope
    if True or show_all:
        mp.plot_maxSS_vs_T(fileroot, RTloop_filenames, '_minSSvsT', Npoints=5, Icutoff=2*10**-12)
    
    
if __name__== "__main__":
  main()

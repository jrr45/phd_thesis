# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:03:54 2020

@author: Justin
"""

#generic imports
import os
import sys
sys.path.append(os.path.join('..','..', 'Code'))
import material_plotter as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator)
from scipy.optimize import fsolve

JR200115_17 = mp.flake_device()
fileroot = r"JR200115_17"
JR200115_17.fileroot = fileroot
JR200115_17.name = "JR200115_17"
JR200115_17.length = 5 * 10**-6 # meters
JR200115_17.width = 30 * 10**-6 # meters, note flake dimensions are weirder than this
JR200115_17.thickness = 13*10**-9 # meters

# R(T) for various gate voltages

RTloop_filenames = [
    'JR200115_17_012_RvsVg_3.0K.txt',
    'JR200115_17_013_RvsVg_50.0K.txt',
    'JR200115_17_014_RvsVg_100.0K.txt',
    'JR200115_17_015_RvsVg_150.0K.txt',
    'JR200115_17_016_RvsVg_200.0K.txt',
    'JR200115_17_017_RvsVg_250.0K.txt',
    'JR200115_17_018_RvsVg_300.0K.txt',
    ]

# ID vs Vg loops at different T, after heating
RTloop_heated_filenames = [
    #'JR200115_17_059_RvsVg_3.0K_rate.txt',
    'JR200115_17_060_RvsVg_3.0K_rate.txt',
    'JR200115_17_061_RvsVg_50.0K_rate.txt',
    'JR200115_17_062_RvsVg_100.0K_rate.txt',
    'JR200115_17_063_RvsVg_150.0K_rate.txt',
    'JR200115_17_064_RvsVg_200.0K_rate.txt',
    'JR200115_17_065_RvsVg_250.0K_rate.txt',
    'JR200115_17_066_RvsVg_300.0K_rate.txt',
    ]

# ID vs Vg loops at different T, again later because
RTloop_after_IV_filenames = [
    'JR200115_17_102_RvsVg_3.0K.txt',
    'JR200115_17_101_RvsVg_50.0K.txt',
    'JR200115_17_100_RvsVg_100.0K.txt',
    'JR200115_17_098_RvsVg_150.0K.txt',
    'JR200115_17_097_RvsVg_200.0K.txt',
    'JR200115_17_096_RvsVg_250.0K.txt',
    'JR200115_17_095_RvsVg_300.0K_rate.txt',  
    ]

# at a higher voltage
RTloop_IV_filenames_100V = [
    'JR200115_17_166_RvsVg_3.0K_redo.txt',
    'JR200115_17_167_RvsVg_10.0K_redo.txt',
    'JR200115_17_168_RvsVg_20.0K_redo.txt',
    'JR200115_17_169_RvsVg_30.0K_redo.txt',
    'JR200115_17_170_RvsVg_40.0K_redo.txt',
    'JR200115_17_171_RvsVg_50.0K_redo.txt',
    'JR200115_17_172_RvsVg_60.0K_redo.txt',
    'JR200115_17_173_RvsVg_70.0K_redo.txt',
    'JR200115_17_174_RvsVg_80.0K_redo.txt',
    'JR200115_17_175_RvsVg_90.0K_redo.txt',
    'JR200115_17_176_RvsVg_100.0K_redo.txt',
    'JR200115_17_177_RvsVg_150.0K_redo.txt',
    'JR200115_17_178_RvsVg_200.0K_redo.txt',
]

Rvst_filenames = [
    'JR200115_17_103_Rvst.txt', #3K
    #'JR200115_17_104_Rvst_3.0K.txt',
    #'JR200115_17_106_Rvst_3.0K.txt',
    'JR200115_17_107_Rvst_3.0K.txt', #3
    #'JR200115_17_163_Rvst_.txt',
    #'JR200115_17_179_Rvst_.txt',
    #'JR200115_17_180_Rvst_.txt',
    'JR200115_17_189_Rvst_.txt', #300K after gating
    ]

MR_sweep_files = [
    'JR200115_17_191_RvsB_100.00Vg_2.0K.txt',
]
    
def get_300K_IDvsVDS_files(figsize=2):
    basename = 'JR200115_17_0'
    start = 67 #23
    device = JR200115_17
    
    filenames = [(basename + str(i) + '_IvsV_300.0K_rate.txt') for i in range(start, start+7)]
    files_pos_inc = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_300.0K_rate.txt') for i in range(start+7, start+14)]
    files_pos_dec = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_300.0K_rate.txt') for i in range(start+14, start+21)]
    files_neg_inc = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_300.0K_rate.txt') for i in range(start+21, start+28)]
    files_neg_dec = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]

    return (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec)

def get_IDvsVDS_3K_files(figsize=2):
    basename = 'JR200115_17_'
    device = JR200115_17
    
    start = 109 #23
    filenames = [(basename + str(i) + '_IvsV_3.0K.txt') for i in range(start, start+7)]
    files_pos_inc = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_3.0K.txt') for i in range(start+7, start+14)]
    files_pos_dec = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_3.0K.txt') for i in range(start+21, start+28)]
    files_neg_inc = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i) + '_IvsV_3.0K.txt') for i in range(start+28, start+35)]
    files_neg_dec = [mp.process_file(os.path.join(device.fileroot, x)) for x in filenames]
    
    return (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec)


# 300K IV plots
def plot_300K_IDvsVDS(figsize=1.5, log=False):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_300K_IDvsVDS_files()

    savename = "JR200115_17_300K_" 
    xadj = 1
    device = JR200115_17
    
    mp.plot_IDvsVDS_generic(device, files_pos_inc, savename + 'positive_increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_pos_dec, savename + 'positive_decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_inc, savename + 'negative_increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_dec, savename + 'negative_decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)
    
def plot_300K_IDVvsVDS_generic(fitfunciton, figsize=2, fitR2=.996, fitpoints=10):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_300K_IDvsVDS_files()
    
    savename = 'JR200115_17_300K_'
    device = JR200115_17
    
    fitfunciton(device, files_pos_inc, savename + '_IDvVDS-positive_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_pos_dec, savename + '_IDvVDS-positive_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_inc, savename + '_IDvVDS-negative_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_dec, savename + '_IDvVDS-negative_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints) 

    
def plot_300K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_FowlerNordheim_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_300K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_PooleFrenkel_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_300K_IDvsVD_Schottky(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVD_Schottky_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_300K_IDvsVDS_SCLC(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVDS_SCLC_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_300K_IDvsVDS_power(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVDS_power_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)
        

# 3K IV plots
def plot_3K_IDvsVDS(figsize=2, log=False):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_IDvsVDS_3K_files()

    savename = '_JR200115_17_3K'
    device = JR200115_17
    xadj = 0
    
    mp.plot_IDvsVDS_generic(device, files_pos_inc, savename + 'positive_increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_pos_dec, savename + 'positive_decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_inc, savename + 'negative_increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_dec, savename + 'negative_decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)


def plot_3K_IDVvsVDS_generic(fitfunciton, figsize=2, fitR2=.996, fitpoints=10):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_IDvsVDS_3K_files()
    
    savename = 'JR200115_17_3K_'
    device = JR200115_17
    
    fitfunciton(device, files_pos_inc, savename + '_IDvVDS-positive_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_pos_dec, savename + '_IDvVDS-positive_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_inc, savename + '_IDvVDS-negative_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_dec, savename + '_IDvVDS-negative_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints) 

def plot_300K_IDVvsVDS_Play(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_Play_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
def plot_300K_IDVvsVDS_DirectTunneling(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_DirectTunneling_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_3K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=.996, fitpoints=10):
    plot_3K_IDVvsVDS_generic(mp.plot_IDVvsVDS_FowlerNordheim_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_3K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=.996, fitpoints=10):
    plot_3K_IDVvsVDS_generic(mp.plot_IDVvsVDS_PooleFrenkel_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_3K_IDvsVD_Schottky(figsize=2, fitR2=.996, fitpoints=10):
    plot_3K_IDVvsVDS_generic(mp.plot_IDvsVD_Schottky_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_IDvsVDS_3K_SCLC(figsize=2, fitR2=.996, fitpoints=10):
    plot_3K_IDVvsVDS_generic(mp.plot_IDvsVDS_SCLC_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_IDvsVDS_3K_power(figsize=2, fitR2=.996, fitpoints=10):
    plot_3K_IDVvsVDS_generic(mp.plot_IDvsVDS_power_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)



def plot_IDvsVg_300K_rate(log, figsize=2, majorx=40):
    colors = mp.colors_set1
    rate_300K_filenames = [
        'JR200115_17_018_RvsVg_300.0K.txt', # .2/.1s = 2 V/s
        'JR200115_17_019_RvsVg_300.0K_rate.txt',# .1V/.1s = 1 V/s
        'JR200115_17_020_RvsVg_300.0K_rate.txt',# .01V/.1s = .1 V/s
        #'JR200115_17_051_RvsVg_300.0K_rate.txt',
        ]

    files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in rate_300K_filenames]
    files = mp.slice_data_each(files, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
        
    mp.plot_IDvsVg_generic(JR200115_17, files, "_JR200115_17_300K_rate", colors, log=log, size=figsize, majorx=40)

def plot_IDvsVg_3K_rate(log=True, figsize=2):
    colors = mp.colors_set1
    
    filenames = ['JR200115_17_010_RvsVg_3.0K.txt', # .2/.1s = 2 V/s
                 'JR200115_17_009_RvsVg_3.0K.txt', # .1V/.1s = 1 V/s
                 'JR200115_17_011_RvsVg_3.0K.txt', # .02/.1s = .2 V/s 
                ]
    
    files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in filenames]
    files = mp.slice_data_each(files, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
        

    mp.plot_IDvsVg_generic(JR200115_17, files, "_JR200115_17_3K_rate", colors, log=True,  size=figsize, majorx=40)

    
def plot_IDvsVg_300K_redo_rate(log=True, size=2, majorx=40):
    colors = mp.colors_set1
    
    filenames = ['JR200115_17_148_RvsVg_300.0K_redo.txt',
                 'JR200115_17_161_RvsVg_300.0K_redo.txt',
                 'JR200115_17_162_RvsVg_300.0K_redo.txt',
                ]
    #colors.append('#FF0000')
    
    files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in filenames]
    files = mp.slice_data_each(files, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)

    mp.plot_IDvsVg_generic(JR200115_17, files, "_JR200115_17_300K_redo_rate", colors, log=log, size=size)


def plot_loopI_cross_section_custom(filenames, savename, increments=[0,25,50,75,100]):
    colors = mp.colors_set1
    colors = [colors[0], colors[3], colors[2], colors[1], colors[4], colors[5]]
              
    #filenames = RTloop_filenames
    files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in filenames]
    
    GVmax = np.nanmax([np.nanmax(file['Gate_Voltage_V']) for file in files])
    index_GVmax = [mp.first_occurance_1D(file['Gate_Voltage_V'], GVmax, starting_index=5)+5 for file in files]

    indexes = []
    for inc in increments:
        indexes.append([mp.first_occurance_1D(file['Gate_Voltage_V'], inc, starting_index=5)+5 for file in files])
    
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{I_{D}}$ (μA)'],
                             yscale='linear', fontsize=10)

    Currents = []
    for fileindexes in indexes:
        current_cross = []
        for (file, index, index_max) in zip(files, fileindexes, index_GVmax):
            if index == index_max:
                step = 4
                GVvals1 = file['Gate_Voltage_V'][(index-step):index:1]
                GVvals2 = file['Gate_Voltage_V'][index:(index+step):1]
                Ivals1 = file['Current_A'][(index-step):index:1]
                Ivals2 = file['Current_A'][index:(index+step):1]
                fit1 = np.poly1d(np.polyfit(GVvals1, Ivals1, 1))
                fit2 = np.poly1d(np.polyfit(GVvals2, Ivals2, 1))

                def f(x):
                    return fit1(x)-fit2(x)
                x = fsolve(f, [100])
                i2add = fit1(x)
            else:
                step = 4
                GVvals = file['Gate_Voltage_V'][(index-step):(index+step):1]
                Ivals = file['Current_A'][index-step:index+step:1]
                fit = np.polyfit(GVvals, Ivals, 1)
                Ifit = np.poly1d(fit)
                i2add = Ifit(file['Gate_Voltage_V'][index])
            current_cross.append(i2add*(10**6))

        Currents.append(current_cross)
    
    temper = [(file['Temperature_K'][0]) for file in files]
    for (color, Is) in zip (colors, Currents):
        ax.plot(temper, Is, '.-', ms=3, linewidth=1.5, color=color)
    
    #ax.set_ylim((None, 3.6))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim((None, 320))
    
    #plt.tight_layout()
    mp.save_generic_svg(fig, JR200115_17, "_loop_I-cross_" + savename)
    
    
def plot_IDSvsVg_effect_heating():
    colors = mp.colors_set1
    
    filenames = ['JR200115_17_052_RvsVg_300.0K.txt',
                 'JR200115_17_054_RvsVg_300.0K_rate.txt',
                 'JR200115_17_056_RvsVg_300.0K_rate.txt',
                 'JR200115_17_058_RvsVg_300.0K_rate.txt',
                ]
    startend = [-75,-75,75,-75]
    
    files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in filenames]
    
    for i, (file, se) in enumerate(zip(files, startend)):
        (files[i],_,_) = mp.slice_data(file, 'Gate_Voltage_V', se, se, .1, starting_index=0)
    
    mp.plot_IDvsVg_generic(JR200115_17, files, '_JR200115_17_effect_heating', colors, log=True, size=2, \
                             majorx=40, ylim=(None,None), fontsize=10, labelsize=10)
    
        
def main(): # sample B
    show_all = False

    #  -- Plot ID vs VG loops --
    figsize = 2
    if False or show_all:
        #mp.plot_IDvsVg_each(JR200115_17.fileroot, RTloop_filenames, '_JR200115_17', log=True, size=figsize, majorx=40,
        #                      ylim=(None,None), fontsize=10, labelsize=10)
        mp.plot_IDvsVg_each(JR200115_17, RTloop_filenames, '_JR200115_17', log=False, size=figsize, majorx=40,
                              ylim=(None,None), fontsize=10, labelsize=10)
    
    # heated ID vs VG loops, and after IV measurements, unused
    if False:
        mp.plot_IDvsVg_each(JR200115_17, RTloop_heated_filenames, '_JR200115_17_heated', log=True, size=figsize, majorx=40,
                               ylim=(None,None), fontsize=10, labelsize=10)
        mp.plot_IDvsVg_each(JR200115_17, RTloop_after_IV_filenames, '_JR200115_17_heated', log=True, size=figsize, majorx=40,
                               ylim=(None,None), fontsize=10, labelsize=10)
        

    # -- Cross section of loop data --
    if False or show_all:
        mp.plot_loopR_cross_section(JR200115_17, RTloop_filenames, "_JR200115_17_RDS", increments=[0,25,50,75],\
                                      figsize=figsize, log=True, xlim=(None, 322), ylim=(None, None))
        mp.plot_loopR_cross_section(JR200115_17, RTloop_filenames[:5], "loop_RDS", increments=[75], figsize=.7,\
                                      log=False, ylim=(None, None), fontsize=8, labelsize=8, xlim=(None, 220), colororder=[1])
        mp.plot_loopR_cross_section(JR200115_17, RTloop_heated_filenames, "_JR200115_17_RDS_heated", increments=[0,25,50,75],\
                                      figsize=figsize, log=True, ylim=(None, 4*10**7))
        mp.plot_loopR_cross_section(JR200115_17, RTloop_after_IV_filenames, "_JR200115_17_RDS_IDVDS", increments=[0,25,50,75], figsize=2,\
                                      log=True, ylim=(None, None), fontsize=10, labelsize=10, xlim=(None, 220))
        mp.plot_loopR_cross_section(JR200115_17, RTloop_IV_filenames_100V, "_JR200115_17_RDS_100V", increments=[0,25,50,75,100], figsize=2,\
                                      log=True, ylim=(None, None), fontsize=10, labelsize=10, xlim=(0, 322))

    
    # -- 3K ID vs VDS curves
    if False:
        mp.plot_IDvVDS_gating_generic(JR200115_17, 'JR200115_17_', '_IvsV_3.0K.txt', 109, 7, "_3K")
    
    # -- 300K ID vs VDS curves
    if False or show_all:
        plot_300K_IDvsVDS(figsize=figsize, log=False)
        plot_300K_IDvsVDS(figsize=figsize, log=True)
        
    # -- 3K ID vs VDS curves
    if False or show_all:
        plot_3K_IDvsVDS(figsize=figsize, log=False)
        plot_3K_IDvsVDS(figsize=figsize, log=True)
        
        # -- Effect off heating sample on ID
    if False or show_all:
        plot_IDSvsVg_effect_heating()

    # -- effect of ramping rate
    if False or show_all:
        plot_IDvsVg_300K_rate(log=True, figsize=figsize)
        #plot_IDvsVg_300K_redo_rate(log=True)
        plot_IDvsVg_3K_rate(log=True, figsize=figsize)
    
    # -- MR, not much data
    if False:
        mp.plot_IDvsB_generic(JR200115_17, MR_sweep_files, '_MR', mp.colors_set1, log=False, symm=True, size=2)
    
    # still working on
    if False:
        files = [mp.process_file(os.path.join(JR200115_17.fileroot, x)) for x in Rvst_filenames[:]]
        mp.plot_IDSvsTime_generic(JR200115_17, files, '_RvsTime', log=False, size=2, majorx=1800, ylim=(None,None))

    # -- carrier mobility μ
    if False or show_all:
        mp.plot_mobility_μ_cross_section(JR200115_17, RTloop_filenames, "_JR200115_17", 
                                         figsize=1.5, ylim=(None, None),\
                                         log=False, increments=[25,50,75], colororder=[3,2,1])
            
    #mp.plot_loopR_cross_section(JR200115_17, RTloop_filenames, "_JR200115_17_RDS", increments=[0,25,50, 75],\
    #                                  figsize=figsize, log=True, ylim=(None, 4*10**7),  colororder=[0,3,2,1,4,5], \
    #                                  fontsize=10, labelsize=8, xlim=(None, 320))
    
    # min subthreshold slope
    if False or show_all:
        mp.plot_maxSS_vs_T(JR200115_17, RTloop_filenames, '_minSSvsT', Npoints=5, Icutoff=1*10**-12)
        mp.plot_maxSS_vs_T(JR200115_17, RTloop_heated_filenames, '_minSSvsT_heated', Npoints=5, 
                           Icutoff=1*10**-12, startend=75, switch=-75.)
        
    # Schotty and Schotty-Simmons for fixed V. ln(J*T^pow) vs 1/T
    if False or show_all:
        mp.plot_Schottky_Simmons_cross_section(JR200115_17, RTloop_filenames, "_JR200115_17", 
                figsize=2, xlim=(0,None),
                ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])
        mp.plot_Schottky_cross_section(JR200115_17, RTloop_filenames, "_JR200115_17", 
                figsize=2, xlim=(0,None),
                ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])
   
    # -- 300K logID vs logVDS curves
    if False or show_all:
        plot_300K_IDvsVDS_SCLC(figsize=figsize)
    # -- 3K logID vs logVDS curves
    if False or show_all:
        plot_IDvsVDS_3K_SCLC(figsize=figsize)
    
    # fit ID vs VDS data    
    if False or show_all:
        fitR2 = .99
        fitpoints=200
        #plot_300K_IDvsVDS_power(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDvsVDS_SCLC(figsize=2, fitR2=fitR2)
        #plot_300K_IDvsVD_Schottky(figsize=2, fitR2=fitR2)
        #plot_300K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=fitR2)
        plot_300K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_DirectTunneling(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_Play(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        
    # fit 3K ID vs VDS data    
    if True or show_all:
        fitR2 = .99
        fitpoints=5
        plot_3K_IDvsVDS(figsize=figsize, log=False)
        #plot_IDvsVDS_3K_power(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_IDvsVDS_3K_SCLC(figsize=2, fitR2=fitR2)
        #plot_3K_IDvsVD_Schottky(figsize=2, fitR2=fitR2)
        #plot_3K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=fitR2)
        #plot_3K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_3K_IDVvsVDS_DirectTunneling(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_3K_IDVvsVDS_Play(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
    
    #mp.plot_play_cross_section(JR200115_17, RTloop_filenames, "JR200115_17", 
    #            figsize=2, xlim=(0,None),
    #            ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])

    
if __name__== "__main__":
  main()


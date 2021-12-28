# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:24:44 2020

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "Nextcloud", "Work", "JR Thesis", "Code"))
import material_plotter as mp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)

import numpy as np
from numpy.lib.recfunctions import append_fields
import numpy.polynomial.polynomial as poly

film_thickness = 45.e-9
# sample A
large_trapazoid_5 = mp.flake_device()
large_trapazoid_5.name = 'large_trapazoid_5'
large_trapazoid_5.fileroot = os.path.join('In-plane', 'large_trapazoid_5')
large_trapazoid_5.thickness = film_thickness
large_trapazoid_5.width = 20.e-6
large_trapazoid_5.volt_length = 38.e-6
large_trapazoid_5.length = 52.e-6

# sample B
small_trapazoid_1 = mp.flake_device()
small_trapazoid_1.name = 'small_trapazoid_1'
small_trapazoid_1.fileroot = os.path.join('In-plane', 'small_trapazoid_1')
small_trapazoid_1.thickness = film_thickness
small_trapazoid_1.width = 20.e-6
small_trapazoid_1.volt_length = 38.e-6
small_trapazoid_1.length = 52.e-6

# sample C
large_trapazoid_8 = mp.flake_device()
large_trapazoid_8.name = 'large_trapazoid_8'
large_trapazoid_8.fileroot = os.path.join('In-plane', 'large_trapazoid_8')
large_trapazoid_8.thickness = 45.e-9
large_trapazoid_8.width = 20.e-6
large_trapazoid_8.volt_length = 38.e-6
large_trapazoid_8.length = 52.e-6

# sample D
large_trapazoid_7 = mp.flake_device()
large_trapazoid_7.name = 'large_trapazoid_7'
large_trapazoid_7.fileroot = os.path.join('In-plane', 'large_trapazoid_7')
large_trapazoid_7.thickness = film_thickness
large_trapazoid_7.width = 20.e-6
large_trapazoid_7.volt_length = 38.e-6
large_trapazoid_7.length = 52.e-6

# sample E
Parallelagram_1 = mp.flake_device()
Parallelagram_1.name = 'Parallelagram_1'
Parallelagram_1.fileroot = os.path.join('In-plane', 'Parallelagram_1')
Parallelagram_1.thickness = film_thickness
Parallelagram_1.width = 20.e-6
Parallelagram_1.volt_length = 38.e-6
Parallelagram_1.length = 52.e-6

# substrate1
substrate1 = mp.flake_device()
substrate1.name = 'GaAs_substrate_1'
substrate1.fileroot = os.path.join('In-plane', 'GaAs_substrate')
substrate1.thickness = 0
substrate1.width = 0.e-6
substrate1.volt_length = 0.e-6
substrate1.length = 0.e-6

# substrate2
substrate2 = mp.flake_device()
substrate2.name = 'GaAs_substrate_2'
substrate2.fileroot = os.path.join('In-plane', 'GaAs_substrate')
substrate2.thickness = 0
substrate2.width = 0.e-6
substrate2.volt_length = 0.e-6
substrate2.length = 0.e-6

combined_samples = [large_trapazoid_5, small_trapazoid_1, large_trapazoid_8,
                    large_trapazoid_7, Parallelagram_1, substrate1, substrate2]

def plot_resistance_vs_T_combined(size=2, colors=mp.colors_set1, log=True, ylim=(None,None)):
    devices = combined_samples
    files = files_RvsT
    
    colors = mp.colors_set1
    filenames = files_RvsT
    
    Resistances = []
    temperatures = []
    # compute resistivities
    for (device, filename, to_plot) in zip(devices, filenames, RvsT_plot):
        if not to_plot:
            continue
        
        files = mp.process_device_files(device, filename)
        temperatures.append(files[0]['Temperature_K'])
        
        #if device == substrate1 or device == substrate2:
        #    Resistances.append(files[0]['Resistance_1_Ohms'])
        #    continue
        
        if device == large_trapazoid_8:
            Resistances.append(files[0]['Resistance_3_Ohms'])
            continue
                  
        ResistanceAVG = (files[0]['Resistance_1_Ohms']+
                         files[0]['Resistance_2_Ohms'])/2
        Resistances.append(ResistanceAVG)
        
    (scale_pow, scale_label) = (1, '')
    if not log:
        ymax = np.nanmax([np.nanmax(y) for y in Resistances]) 
        (scale_pow, scale_label) = mp.m_order(ymax)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=['$\it{T}$ ($K$)', '$\it{R}$ (%sΩ)' % scale_label],
                        yscale=('log' if log else 'linear'))
    
    for (Xdata, Ydata, color) in zip(temperatures, Resistances, colors):
        ind = mp.first_occurance_1D(Xdata, 300)
        if ind is not None:
            print(device.name + " resistance at 300K: " + str(Ydata[ind]))
        ax.plot(Xdata, (Ydata if log else Ydata*scale_pow),
            ',-', ms=3, linewidth=1.5, color=color)
    
    ax.set_xlim((294, 406))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    #ax.set_ylim((4*10**6, 1.5*10**9))
    
    # save
    device = mp.device()
    device.name = 'combined'
    mp.save_generic_svg(fig, device, '_RvsT_'+('log' if log else 'linear'))
    plt.show() 
    plt.clf() # no need to keep this in memory
    return None

def plot_rho_vs_T_combined(size=2, colors=mp.colors_set1, log=True, ylim=(None,None)):
    devices = combined_samples
    files = files_RvsT
    
    colors = mp.colors_set1
    filenames = files_RvsT
    
    
    rhos = []
    temperatures = []
    # compute resistivities
    for (device, filename, to_plot) in zip(devices, filenames, RvsT_plot):
        if not to_plot:
            continue
        
        files = mp.process_device_files(device, filename)
        temperatures.append(files[0]['Temperature_K'])
        
        if device == large_trapazoid_8:
            rhos.append(files[0]['Resistance_3_Ohms'] * device.width * \
                        device.thickness / device.volt_length)
            continue
                  
        
        Resistance1 = files[0]['Resistance_1_Ohms']
        Resistance2 = files[0]['Resistance_2_Ohms']
        rho1 = Resistance1 * device.width * device.thickness / device.volt_length
        rho2 = Resistance2 * device.width * device.thickness / device.volt_length
        rhoAVG = (rho1+rho2)/2
        rhos.append(rhoAVG)
        
    (scale_pow, scale_label) = (1, '')
    if not log:
        ymax = np.nanmax([np.nanmax(y) for y in rhos]) 
        (scale_pow, scale_label) = mp.m_order(ymax)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=['$\it{T}$ ($K$)', '$\it{ρ}$ (%sΩ$\cdot$cm)' % scale_label],
                        yscale=('log' if log else 'linear'))
    
    for (Xdata, Ydata, color) in zip(temperatures, rhos, colors):
        ind = mp.first_occurance_1D(Xdata, 300)
        if ind is not None:
            print(device.name + " resitvity at 300K: " + str(Ydata[ind]))
        ax.plot(Xdata, (Ydata if log else Ydata*scale_pow),
            ',-', ms=3, linewidth=1.5, color=color)
    
    ax.set_xlim((294, 406))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    
    
    # save
    device = mp.device()
    device.name = 'combined'
    mp.save_generic_svg(fig, device, '_rhovsT_'+('log' if log else 'linear'))
    plt.show() 
    plt.clf() # no need to keep this in memory
    return None

def get_R4pt(device, temperature, rawdata=True):
    ind = combined_samples.index(device)
    Ilimits = (None, None)
    fields = ['Voltage_1_V', 'Voltage_2_V']
    
    if temperature == 300:
        if device == large_trapazoid_5 or device == large_trapazoid_7:
            return 0
        
        IV_file = IV_files_300K[ind]
        IV_data = mp.process_device_files(device, IV_file) 
        
        if device == small_trapazoid_1:
            Ilimits = (10e-11, None)
            
        if device == large_trapazoid_8:
            Ilimits = (-.95e-9, 0)
            fields = ['Voltage_1_V']
            IV_data[0]['Voltage_1_V'] = -IV_data[0]['Voltage_1_V']
        if device == Parallelagram_1:
            Ilimits = (-5e-10, None)
        
    if temperature == 400:
        IV_file = IV_files_400K[ind]
        IV_data = mp.process_device_files(device, IV_file) 
        
        if device == small_trapazoid_1:
            IV_data[0]['Voltage_1_V'] = -IV_data[0]['Voltage_1_V']
        if device == large_trapazoid_8:
            IV_data[0]['Voltage_1_V'] = -IV_data[0]['Voltage_1_V'] 
        if device == large_trapazoid_7:
            IV_data[0]['Voltage_2_V'] = -IV_data[0]['Voltage_2_V'] 
            Ilimits = (-2e-10, None)
    (R4pt, _, _) = mp.process_IV_data(device, IV_data[0], fields,
                                  Ilimits=Ilimits, plot_data=rawdata)
    return R4pt
      
files_RvsT = [
    'large_trapazoid_5_007_RvsT_4pt.txt',
    'small_trapazoid_1_012_RvsT_4pt.txt',
    'large_trapazoid_8_011_RvsT_4pt.txt',
    'large_trapazoid_7_005_RvsT.txt',
    'Parallelagram_1_010_RvsT_4pt.txt',
    'pair7-4pt_006_RvsT_bottom-top.txt',
    'pair7-4pt_005_RvsT_bottom_outliers_removed.txt',
    ]
Hall_files_400K = [
    'large_trapazoid_5_003_RvsB_400.0K.txt',
    'small_trapazoid_1_016_RvsB_400.0K.txt',
    'large_trapazoid_8_001_RvsB_400.0K.txt',
    'large_trapazoid_7_007_RvsB_400.0K.txt',
    'Parallelagram_1_006_RvsB_400.0K.txt',
    ]
IV_files_400K = [
    'large_trapazoid_5_008_VvsI_400.0K.txt',
    'small_trapazoid_1_015_VvsI_400.0K.txt',
    'large_trapazoid_8_002_VvsI_400.0K.txt',
    'large_trapazoid_7_006_VvsI_400.0K_4pt.txt',
    'Parallelagram_1_007_VvsI_400.0K_4pt.txt'
    ]
Hall_files_300K = [
    '',
    'small_trapazoid_1_010_RvsB_300.0K.txt',
    'large_trapazoid_8_004_RvsB_300.0K.txt', 
    '',
    'Parallelagram_1_004_RvsB_300.0K.txt',
    ]
IV_files_300K = [
    '',
    'small_trapazoid_1_011_VvsI_300.0K_4pt.txt',
    'large_trapazoid_8_003_VvsI_300.0K.txt', #flipped
    '',
    'Parallelagram_1_003_IvsV_300.0K.txt'
    ]
RvsT_plot = [
    True, # good
    True, # good
    False, # 2pt only
    False, # jagged
    False, # small spike
    True, #substrate
    True, #substrate
    ]
Hall_plot_300K = [
    (False, False), # no data
    (True, False), # good, good
    (False, True), # bad, good
    (False, False), # no data
    (False, False), #probably not a good idea
    ]
Hall_plot_400K = [
    (True, False), # good, good
    (False, False), # both weirdly flat, s-like when zoomed in
    (False, True), # very S-like, kind of s-like, check for MR
    (False, False), # spiky at 5+, spike at 2.5 then back to normal
    (False, False), #probably not a good idea
    ]

def plot_VH_vs_H_combined(size=2, colors=mp.colors_set1, rawdata=False, log=False, fit=True, 
                               Brange=(-5, 5), Bfitlimits=(-2, 2), temperature=400,
                               ylim=(None,None)):
    devices = combined_samples
    if temperature == 400:
        Hall_files = Hall_files_400K
        IV_files = IV_files_400K
        Hall_plots = Hall_plot_400K
    if temperature == 300:
        Hall_files = Hall_files_300K
        IV_files = IV_files_300K
        Hall_plots = Hall_plot_300K
    
    ydatas = []
    ymaxes = []
    magnetic_fields = []
    
    for (device, Hall_file, IV_file, to_plot) in \
        zip(devices, Hall_files, IV_files, Hall_plots): 
        if not to_plot[0] and not to_plot[1]:
            ydatas.append(np.empty((0,0)))
            magnetic_fields.append(np.empty((0,0)))
            continue
        
        R4pt = get_R4pt(device, temperature, rawdata=rawdata)
        #IV_data = mp.process_device_files(device, IV_file) 
        #(R4pt, _, _) = mp.process_IV_data(device, IV_data[0], ['Voltage_1_V', 'Voltage_2_V'],
        #                                  Ilimits=Ilimit, plot_data=rawdata)
        
        print('resistance = ' + str(R4pt))
        Hall_data = mp.process_device_files(device, Hall_file)
        mp.slice_data_each(Hall_data, "Magnetic_Field_T", Brange[0], Brange[1], .005, nth_start=1, 
                           nth_finish=1, starting_index=0)
        
        #symmetrize and fit hall data
        (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
            mp.process_hall_data(device, Hall_data[0], T_Rxx_4pt=R4pt,
                                 hall_fields=['Voltage_1_V', 'Voltage_2_V'],#['Voltage_1_V', 'Voltage_2_V'], 
                                 symmeterize=True, Bfitlimits=Bfitlimits)
        
        #append symetrized data, fit data
        current = Hall_data[0]['Current_A']
        ydatas.append((VH_datas[0]/current, VH_datas[1]/current,
                       fitdata[0]/current, fitdata[1]/current))
        magnetic_fields.append(B_data)
        
        if (to_plot[0]):
            ymaxes.append(np.nanmax([np.nanmax(y) for y in VH_datas[0]/current]))
        if (to_plot[1]):
            ymaxes.append(np.nanmax([np.nanmax(y) for y in VH_datas[1]/current]))
    
    
    #plots
    scale_pow = 1.
    scale_label = ''
    if not log:
        (scale_pow, scale_label) = mp.m_order(ymaxes)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=['$\it{B}$ (T)', '$\it{R_{H}}$ (%sΩ)' % scale_label],
                               yscale=('log' if log else 'linear'))
    
    i = 0
    for (magnetic_field, V_H, to_plot) in zip(magnetic_fields, ydatas, Hall_plots):
        if to_plot[0]:
            ax.plot(magnetic_field, scale_pow*V_H[0], '.-', ms=3, linewidth=1.5, color=colors[i])
            i = i + 1
            
            if fit:
                ax.plot(magnetic_field, scale_pow*V_H[2], '-', ms=0, linewidth=1., color='black')
                
        if to_plot[1]:
            ax.plot(magnetic_field, scale_pow*V_H[1], '.-', ms=3, linewidth=1.5, color=colors[i])
            i = i + 1
            
            if fit:
                ax.plot(magnetic_field, scale_pow*V_H[3], '-', ms=0, linewidth=1., color='black')
    
    ax.set_xlim(Brange)
    ax.set_ylim(ylim)
    
    # save
    device = mp.device()
    device.name = 'combined'
    mp.save_generic_svg(fig, device, '_RHvsB_' + str(temperature) + 'K_' \
                        +('log' if log else 'linear'))
    plt.show() 
    plt.clf() # no need to keep this in memory
    return None

def main(): 
    colors=mp.colors_set1
    
    plot_resistance_vs_T_combined(size=2, log=True, colors=colors)
    #plot_rho_vs_T_combined(size=2, log=True, colors=colors)
    Bmax = 2
    #plot_VH_vs_H_combined(size=2, colors=[colors[1], colors[2]], temperature=300, 
    #                      Brange=(-Bmax, Bmax), Bfitlimits=(-2, 2))
    #plot_VH_vs_H_combined(size=2, colors=[colors[0], colors[2]], 
    #                      temperature=400, 
    #                      Brange=(-Bmax, Bmax), Bfitlimits=(-2, 2))
    #get_R4pt(combined_samples[0], temperature=300, rawdata=True)
if __name__== "__main__":
  main()
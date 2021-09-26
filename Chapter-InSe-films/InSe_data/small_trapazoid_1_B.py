# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:43:09 2020

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "Nextcloud", "Work", "JR Thesis", "Code"))
import material_plotter as mp

import numpy as np
from numpy.lib.recfunctions import append_fields
import numpy.polynomial.polynomial as poly

small_trapazoid_1 = mp.flake_device()
small_trapazoid_1.name = 'small_trapazoid_1'
small_trapazoid_1.fileroot = os.path.join('In-plane', 'small_trapazoid')
small_trapazoid_1.thickness = 42.e-9
small_trapazoid_1.width = 20.e-6
small_trapazoid_1.volt_length = 38.e-6
small_trapazoid_1.length = 52.e-6


def plot_rho_vs_T_fits(size=2):
    device = small_trapazoid_1
    filename = 'small_trapazoid_1_012_RvsT_4pt.txt'
    files = mp.process_device_files(device, filename)
    Tmin = 310
    R1 = True #red
    R2 = True #blue
    
    mp.plot_rho_vs_T_hopping_generic(device, files[0], power=(-1/4), power_label='-1/4',
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_rho_vs_T_hopping_generic(device, files[0], power=(-1/3), power_label='-1/3',
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_rho_vs_T_hopping_generic(device, files[0], power=(-1/2), power_label='-1/2',
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_rho_vs_T_intrinsic_generic(device, files[0],
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_rho_vs_T_hopping_generic(device, files[0], power=(1.), power_label='1',
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_WvsT_fit_generic(device, files[0], 'wvsT',
                            size=size, Tmin=Tmin, R1=R1, R2=R2)
    
    mp.plot_rho_vs_T_power_generic(device, files[0], Tmin=Tmin, R1=R1, R2=R2)

def plot_rho_vs_T(size=2, log=True, power=(1), power_label='1'):
    device = small_trapazoid_1
    filename = 'small_trapazoid_1_012_RvsT_4pt.txt'
    files = mp.process_device_files(device, filename)
    file = files[0]
    file['Resistance_2_Ohms'][:5] = np.nan
    
    mp.plot_rho_vs_T_generic(device, file, device.name, size=size, log=log, 
                             power=power, power_label=power_label, xlim=(300, 407),
                             R1=False, R2=False, RDS=False, RAVG=True)
    
def plot_MR_vs_H_400K(size=2, rawdata=False):
    device = small_trapazoid_1
    
    MR_files = mp.process_device_files(device, 'small_trapazoid_1_017_RvsB_400.0K.txt') 
    MR_file = MR_files[0]
    MR_file['Magnetic_Field_T'][0] = np.nan
    MR_file['Voltage_1_V'][0] = np.nan
    MR_file['Voltage_2_V'][0] = np.nan
    MR_file['Voltage_3_V'][0] = np.nan
    mp.process_MR_data(small_trapazoid_1, MR_file, ['Voltage_1_V', 'Voltage_2_V'], Bfitlimits=(None,None), plot_data=True, fit_data=False)
 

def plot_VH_vs_H_400K(size=2, rawdata=False):
    device = small_trapazoid_1
    
    RvsBfile = 'small_trapazoid_1_016_RvsB_400.0K.txt'
    
    # use IV data to compute 4pt resistance
    VvsIfile = 'small_trapazoid_1_015_VvsI_400.0K.txt' 
    IV_files = mp.process_device_files(device, VvsIfile) 
    IV_file = IV_files[0]
    IV_file['Voltage_1_V'] = -IV_file['Voltage_1_V'] #leads were backwards
    (R4pt, _, _) = mp.process_IV_data(device, IV_file, ['Voltage_1_V', 'Voltage_2_V'],
                                      Ilimits=(-30e-9,None), plot_data=rawdata)

    #Load hall data
    Hall_files = mp.process_device_files(device, RvsBfile) 
    Hall_file = Hall_files[0]
    
    #symmetrize and fit hall data
    (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
        mp.process_hall_data(small_trapazoid_1, Hall_file, T_Rxx_4pt=R4pt,
                             hall_fields=['Voltage_1_V', 'Voltage_2_V'], 
                             symmeterize=True, Bfitlimits=(-2.0,2.0))

    #append symetrized data, fit data
    Hall_file['Magnetic_Field_T'] = B_data
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_1_V', VH_datas[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_2_V', VH_datas[1], np.float64, usemask=False)
    
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_1_V', fitdata[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_2_V', fitdata[1], np.float64, usemask=False)
    
    
    #plots
    colors = [mp.colors_set1[0], mp.colors_set1[1], 'black', 'black']
    
    if rawdata:
        mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                          ['Voltage_1_V','Voltage_2_V'],
                          '$\it{V_{H}}$ (%sV)', '_VHvsB_raw', markers=['.-','.-','-','-'], 
                          device=small_trapazoid_1, files=[Hall_file], savename=device.name+'_VHvsB_raw',
                          colors=mp.colors_set1, log=False, size=size)
        
        mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                          ['Voltage_3_V'],
                          '$\it{V_{SD}}$ (%sV)', '_VvsB_sd', markers=['.-','.-','-','-'], 
                          device=small_trapazoid_1, files=[Hall_file], savename=device.name+'_VvsB_sd', 
                          colors=[mp.colors_set1[2]], log=False, size=size)
    
    mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                      ['Hall_Voltage_1_V','Hall_Voltage_2_V', 
                       'Hall_Voltage_fit_1_V', 'Hall_Voltage_fit_2_V'],
                      '$\it{V_{H}}$ (%sV)', '_VHvsB_', markers=['.-','.-','-','-'], 
                      #xlim=(-3,3), ylim=(-700,700),majorx=1,
                      device=small_trapazoid_1, files=[Hall_file], 
                      savename=device.name+'_VHvsB_400K',
                      colors=colors, log=False, size=size)

def plot_VH_vs_H_300K(size=2, rawdata=False):
    device = small_trapazoid_1
    
    RvsBfile = 'small_trapazoid_1_010_RvsB_300.0K.txt'
    VvsIfile = 'small_trapazoid_1_011_VvsI_300.0K_4pt.txt'
    IV_files = mp.process_device_files(device, VvsIfile) 
    IV_file = IV_files[0]
    (R4pt, _, _) = mp.process_IV_data(device, IV_file, ['Voltage_1_V', 'Voltage_2_V'],
                                      Ilimits=(10e-11,None), plot_data=rawdata)
    #R4pt = 5.1789e+08
    
    Hall_files = mp.process_device_files(device, RvsBfile) 
    Hall_file = Hall_files[0] 
    
    #symmetrize and fit hall data
    (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
        mp.process_hall_data(small_trapazoid_1, Hall_file, T_Rxx_4pt=R4pt,
                             hall_fields=['Voltage_1_V', 'Voltage_2_V'], 
                             symmeterize=True, Bfitlimits=(-2.0,2.0))

    #append symetrized data, fit data
    Hall_file['Magnetic_Field_T'] = B_data
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_1_V', VH_datas[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_2_V', VH_datas[1], np.float64, usemask=False)
    
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_1_V', fitdata[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_2_V', fitdata[1], np.float64, usemask=False)
    
    
    colors = [mp.colors_set1[0],mp.colors_set1[1], 'black', 'black']
    
    if rawdata:
        mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                          ['Voltage_1_V','Voltage_2_V'],
                          '$\it{V_{H}}$ (%sV)', '_VHvsB_raw', markers=['.-','.-','-','-'], 
                          device=small_trapazoid_1, files=[Hall_file], savename='_VHvsB_raw', colors=mp.colors_set1, log=False, size=size)
        
        mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)', ['Voltage_3_V'],
                          '$\it{V_{SD}}$ (%sV)', '_VvsB_sd', markers=['.-','.-','-','-'], 
                          device=small_trapazoid_1, files=[Hall_file], savename='_VvsB_sd', colors=[mp.colors_set1[2]], log=False, size=size)
    
    mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                      ['Hall_Voltage_1_V','Hall_Voltage_2_V', 
                       'Hall_Voltage_fit_1_V', 'Hall_Voltage_fit_2_V'],
                      '$\it{V_{H}}$ (%sV)', '_VHvsB_', markers=['.-','.-','-','-'], 
                      device=small_trapazoid_1, files=[Hall_file], 
                      savename=device.name+'_VHvsB_300K',
                      colors=colors, log=False, size=size)

IDvsVDS_400K_files = [
    #'small_trapazoid_1_008_VvsI_400.0K.txt',
    'small_trapazoid_1_014_VvsI_400.0K.txt',
    'small_trapazoid_1_015_VvsI_400.0K.txt'
    ]

IDvsVDS_300K_files = [
    'small_trapazoid_1_006_IvsV.txt',
    #'small_trapazoid_1_011_VvsI_300.0K_4pt.txt',
    ]

def plot_IDvsVDS(size=2):
    device = small_trapazoid_1
    colors = mp.get_IDvsVDS_colors()
    
    IV_files = mp.process_device_files(device, IDvsVDS_300K_files)
    mp.plot_IDvsVDS_generic(device, IV_files, '_300K', colors, size=size)
    
    IV_files = mp.process_device_files(device, IDvsVDS_400K_files)
    mp.plot_IDvsVDS_generic(device, IV_files, '_400K', colors, size=size)

def main(): 
    show_all = False
    
    plot_rho_vs_T(size=2, log=False)
    plot_VH_vs_H_300K(size=2)
    plot_VH_vs_H_400K(size=2)
    #plot_IDvsVDS(size=2)
    
    #plot_rho_vs_T_fits(size=2)
    
if __name__== "__main__":
  main()
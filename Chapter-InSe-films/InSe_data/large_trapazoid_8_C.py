# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:32:13 2020

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "Nextcloud", "Work", "JR Thesis", "Code"))
import material_plotter as mp

import numpy as np
from numpy.lib.recfunctions import append_fields
import numpy.polynomial.polynomial as poly

large_trapazoid_8 = mp.flake_device
large_trapazoid_8.name = 'large_trapazoid_8'
large_trapazoid_8.fileroot = os.path.join('In-plane', 'large_trapazoid_8')
large_trapazoid_8.thickness = 45.e-9
large_trapazoid_8.width = 20.e-6
large_trapazoid_8.volt_length = 38.e-6
large_trapazoid_8.length = 52.e-6


def plot_rho_vs_T_fits(size=2, fitpoints=10):
    device = large_trapazoid_8
    filename = 'large_trapazoid_8_011_RvsT_4pt.txt'
    files = mp.process_device_files(device, filename)
    Tmin = 0
    R1 = False
    R2 = False
    
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
    device = large_trapazoid_8
    filename = 'large_trapazoid_8_011_RvsT_4pt.txt'
    files = mp.process_device_files(device, filename)
    file = files[0]
    
    mp.plot_rho_vs_T_generic(device, file, device.name, size=size, log=log, 
                             power=power, power_label=power_label, xlim=(300, 407),
                             R1=False, R2=False, RDS=True, RAVG=False)



def plot_VH_vs_H_400K(size=2):
    device = large_trapazoid_8
    Hall_files = mp.process_device_files(device, 'large_trapazoid_8_001_RvsB_400.0K.txt')
    IV_files = mp.process_device_files(device, 'large_trapazoid_8_002_VvsI_400.0K.txt') 
    IV_file = IV_files[0]
    Hall_file = Hall_files[0]
    IV_file['Voltage_1_V'] = -IV_file['Voltage_1_V']
    
    (R4pt, _, _) = mp.process_IV_data(device, IV_file, ['Voltage_1_V', 'Voltage_2_V'], Ilimits=(None,None))
    
    (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
        mp.process_hall_data(device, Hall_file, T_Rxx_4pt=R4pt,
                             hall_fields=['Voltage_1_V', 'Voltage_2_V'],
                             symmeterize=True, Bfitlimits=(-2,2))

    Hall_file['Magnetic_Field_T'] = B_data
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_1_V', VH_datas[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_2_V', VH_datas[1], np.float64, usemask=False)
    
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_1_V', fitdata[0], np.float64, usemask=False)
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_2_V', fitdata[1], np.float64, usemask=False)
    
    
    colors = [mp.colors_set1[0],mp.colors_set1[1], 'black', 'black']
    
    mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                      ['Hall_Voltage_1_V','Hall_Voltage_2_V', 
                       'Hall_Voltage_fit_1_V', 'Hall_Voltage_fit_2_V'],
                      '$\it{V_{H}}$ (%sV)', '_VHvsB_', markers=['.-','.-','-','-'],
                      device=device, files=[Hall_file], 
                      savename=device.name+'_VHvsB_400K',
                      colors=colors, log=False, size=size)

def plot_VH_vs_H_300K(size=2):
    device = large_trapazoid_8
    Hall_files = mp.process_device_files(device, 'large_trapazoid_8_004_RvsB_300.0K.txt')
    IV_files = mp.process_device_files(device, 'large_trapazoid_8_003_VvsI_300.0K.txt') 
    IV_file = IV_files[0]
    Hall_file = Hall_files[0]
    IV_file['Voltage_1_V'] = -IV_file['Voltage_1_V']
    
    (R4pt, _, _) = mp.process_IV_data(device, IV_file, ['Voltage_1_V'], Ilimits=(-.95e-9,0))

    B_data = Hall_file['Magnetic_Field_T']
    occ0 = mp.first_occurance_1D(B_data, -6.0, tol=.05, starting_index=0)
    occ1 = mp.first_occurance_1D(B_data, 6.0, tol=.05, starting_index=0)
    B_data[:occ0] = np.nan
    B_data[occ1+1:] = np.nan
    Hall_file['Magnetic_Field_T'] = B_data
    
    (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
        mp.process_hall_data(device, Hall_file, T_Rxx_4pt=R4pt,
                             hall_fields=['Voltage_2_V'], symmeterize=True,
                             Bfitlimits=(-2,2))


    Hall_file['Magnetic_Field_T'] = B_data
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_2_V', VH_datas[0], 
                              np.float64, usemask=False)
    
    Hall_file = append_fields(Hall_file, 'Hall_Voltage_fit_2_V', fitdata[0],
                              np.float64, usemask=False)
    
    
    colors = [mp.colors_set1[1], 'black', 'black']
    
    mp.plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                      ['Hall_Voltage_2_V', 'Hall_Voltage_fit_2_V'],
                      '$\it{V_{H}}$ (%sV)', '_VHvsB_', markers=['.-','-','-'],
                      device=device, files=[Hall_file],
                      savename=device.name+'_VHvsB_300K',
                      colors=colors, log=False, size=size)

def main(): 
    show_all = False
    
    #plot_rho_vs_T(size=2, log=False)
    #plot_VH_vs_H_400K(size=2)
    plot_VH_vs_H_300K(size=2)
    
    #plot_rho_vs_T_fits(size=2)
    
if __name__== "__main__":
  main()
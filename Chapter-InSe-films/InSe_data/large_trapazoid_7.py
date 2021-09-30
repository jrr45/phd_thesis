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

large_trapazoid_7 = mp.flake_device
large_trapazoid_7.name = 'large_trapazoid_7'
large_trapazoid_7.fileroot = os.path.join('In-plane', 'large_trapazoid')
large_trapazoid_7.thickness = 42.e-9
large_trapazoid_7.width = 20.e-6
large_trapazoid_7.volt_length = 38.e-6
large_trapazoid_7.length = 52.e-6


def plot_rho_vs_T_fits(size=2, fitpoints=10):
    device = large_trapazoid_7
    filename = 'large_trapazoid_7_005_RvsT.txt'
    files = mp.process_device_files(device, filename)
    Tmin = 363
    R1 = True
    R2 = True
    
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

def plot_rho_vs_T(size=2, power=(-1/3), power_label='-1/3'):
    device = large_trapazoid_7
    filename = 'large_trapazoid_7_005_RvsT.txt'
    files = mp.process_device_files(device, filename)
    newfiles = []
    
    for file in files:
        
        R1 = file['Resistance_1_Ohms']
        R2 = file['Resistance_2_Ohms']
        
        Ravg = (R1 + R2)/2
        
        rho = Ravg * device.width * device.thickness / device.volt_length
        
        lnrho = np.log(rho*100)
        
        Tpow = np.power(file['Temperature_K'], power)
        
        file = append_fields(file, 'LnResistance_LnOhms', lnrho, np.double, usemask=False)
        file = append_fields(file, 'PowTemperature_K', Tpow, np.double, usemask=False)
        newfiles.append(file)
        
        lnrho = lnrho[~np.isnan(Ravg)]
        Tpow = Tpow[~np.isnan(Ravg)]
        
        coeff, stats = poly.polyfit(Tpow, lnrho, 1, full = True)
        print("slope: %5f, Intercept: %8f, T0: %8.8f K" % (coeff[1], coeff[0], np.power(coeff[1],-1./power)))
        lnRfit = poly.polyval(Tpow, coeff)
        r2_score, sse, tse = mp.compute_r2_weighted(lnrho, lnRfit)
        print("R^2: %5f" % r2_score)
        
        def vrhfit(x, R0, T0):
            return R0 * np.exp((T0 / x)**-power)
        

    clean_label = power_label.replace(r"/", "_") 
    mp.plot_YvsX_generic('PowTemperature_K', '$\it{T^{%s}}$ ($K^{%s}$)' % (power_label, power_label),
                       ['Resistance_1_Ohms','Resistance_2_Ohms', 'Resistance_3_Ohms'], '$\it{R}$ (Ω)', '_play' + clean_label,
                       device=device, files=newfiles, savename='_play', colors=mp.colors_set1, log=True, size=size)


    mp.plot_YvsX_generic('PowTemperature_K', '$\it{T^{%s}}$ ($K^{%s}$)' % (power_label, power_label),
                      'LnResistance_LnOhms', '$\it{ln(ρ)}$ (ln(Ω$\cdot$cm))', '_logRSDvsT_' + clean_label,
                      device=device, files=newfiles, savename='lnrho_vs_T', colors=mp.colors_set1, log=False, size=size)

def plot_VH_vs_H_400K(size=2):
    device = large_trapazoid_7
    Hall_files = mp.process_device_files(device, 'large_trapazoid_7_007_RvsB_400.0K.txt')
    IV_files = mp.process_device_files(device, 'large_trapazoid_7_006_VvsI_400.0K_4pt.txt') 
    IV_file = IV_files[0]
    Hall_file = Hall_files[0]
    IV_file['Voltage_1_V'] = -IV_file['Voltage_1_V']
    
    (R4pt, _, _) = mp.process_IV_data(device, IV_file, ['Voltage_1_V', 'Voltage_2_V'], Ilimits=(None,None))

    (B_data, VH_datas, n2Ds, fits, fitdata, r_squared, μH) = \
        mp.process_hall_data(device, Hall_file, T_Rxx_4pt=R4pt,
                             hall_fields=['Voltage_1_V', 'Voltage_2_V'], symmeterize=True, Bfitlimits=(-1.4,1.4))

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

def main(): 
    show_all = False
    
    plot_rho_vs_T(size=2)
    plot_VH_vs_H_400K(size=2)
    
    #plot_rho_vs_T_fits(size=2)
    
if __name__== "__main__":
  main()
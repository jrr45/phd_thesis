# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:43:09 2020

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "Nextcloud", "Work", "JR Thesis", "Code"))
import material_plotter as mp
import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize

import numpy as np
from numpy.lib.recfunctions import append_fields
import numpy.polynomial.polynomial as poly
from matplotlib.ticker import (MultipleLocator)

small_trapazoid_1 = mp.flake_device()
small_trapazoid_1.name = 'small_trapazoid_1'
small_trapazoid_1.fileroot = os.path.join('In-plane', 'small_trapazoid_1')
small_trapazoid_1.thickness = 45.e-9
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
    fields = ['Voltage_1_V', 'Voltage_2_V']
    
    MR_files = mp.process_device_files(device, 'small_trapazoid_1_017_RvsB_400.0K.txt') 
    MR_file = MR_files[0]
    MR_file['Magnetic_Field_T'][0] = np.nan
    MR_file['Voltage_1_V'][0] = np.nan
    MR_file['Voltage_2_V'][0] = np.nan
    MR_file['Voltage_3_V'][0] = np.nan
    
    ind = mp.first_occurance_1D(MR_file['Magnetic_Field_T'], 0, 
                                tol=0.2, starting_index=0)

    mp.process_MR_data(small_trapazoid_1, MR_file, fields, 
                       Bfitlimits=(None,None), plot_data=True, 
                       fit_data=False)

# Reuse WL expression
def F(Bx, B):
    return np.log(Bx/B) - special.digamma(.5 + (Bx/B))
def F2(Bx, B):
    return np.log(B/Bx) + special.digamma(.5 + 1./(B/Bx))
def F3(Bx, B):
    return np.log(Bx/B) + special.digamma(.5 + 1./(Bx/B))

#pack as vector
def WL_wiki2(B, Bx):
    (Bϕ, BSO, Be) = (Bx[0], Bx[1], B[2])
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
# full WL formula
def WL_wiki(B, Bϕ, BSO, Be):
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI

def MR_SC_fit(B, mu):
    return 1./(1+(mu*B)**2)  #WIKI


def plot_σMCvsB_custom(color_order=[0,1,2,3,4,5], log=False, \
                     fit_lim=np.inf, size=2, fontsize=10, labelsize=10,
                     xmult=4, fit=False):    
    device = small_trapazoid_1
    filenames =['small_trapazoid_1_017_RvsB_400.0K.txt',]
    savename = '_magnetoconductance_' 
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    colors = mp.colors_set1
    colors = colors[color_order]
    
    files = mp.process_device_files(device, filenames)
        
    file = files[0]
    file = file[1:-1] #first value is bad
    
    H = file['Magnetic_Field_T']
    R1 = file['Resistance_1_Ohms']
    R2 = file['Resistance_2_Ohms']
    Temperature_K = file['Temperature_K'][0]
    
    #symmetrize it
    R1 = (R1+R1[::-1])/2
    R2 = (R2+R2[::-1])/2
    
    Rsq1 = mp.sheet_resistance(device, R1)
    Rsq2 = mp.sheet_resistance(device, R2)
    
    #print(H)
    Rsq_01 = Rsq1[np.argmin(np.abs(H))]
    Rsq_02 = Rsq2[np.argmin(np.abs(H))] 
    
    # deviation from σ(H=0)
    Δσ1 = -(Rsq1-Rsq_01)/(Rsq_01**2)
    Δσ2 = -(Rsq2-Rsq_02)/(Rsq_02**2)

    max1 = np.nanmax(np.abs(Δσ1))
    max2 = np.nanmax(np.abs(Δσ2))
    maxall = max(max1, max2)
    (scale_pow, scale_label) = mp.m_order(maxall)
    if log:
        scale_pow = 1
        scale_label = ''
    
    ax = mp.pretty_plot_single(fig, labels=["B (T)", 'Δσ (%sS)' % scale_label],
                         yscale=('log' if log else 'linear'), 
                         fontsize=fontsize, labelsize=labelsize)    
    ax.plot(H, scale_pow*Δσ1, '.-', ms=3, linewidth=1., color=mp.colors_set1[1])
    #ax.plot(H, scale_pow*Δσ2, '.-', ms=3, linewidth=1., color=mp.colors_set1[1])

    
    if fit and Temperature_K < fit_lim:
        Δσ = Δσ1
        #fit stuff
        ind = np.where(H > 0)
        H_half = H[ind]
        Δσ  = Δσ[ind]
                
        bounds = ([0,0,0],1000)
        #bounds = [0,1000],[0,1000],[0,1000]]
        best_r2 = 0
        best_popt = [1,1,1]
        
        # run multiple times and take the best one
        for i in range(1,10):
            #result = annealing.curve_fit(WL_wiki2, H, Δσ, bounds=bounds)

            #popt = result.x # optimal fit parameters
            
            p0 = np.ndarray.flatten(np.random.rand(1,1))
            #p0[3] = p0[3] - .5
            p0 = np.multiply(p0,.0001*i)
            #p0 = [0.20277506744303325, 7.959480841631488e-07, 0.2043696078628539]
            
            popt, pcov = optimize.curve_fit(MR_SC_fit, H_half, Δσ, maxfev=10000, ftol=10**-15, \
                                            xtol=10**-15, gtol=10**-15, bounds=bounds, p0=p0)
            Δσfit = WL_wiki(H_half, *popt)
            # residual sum of squares
            ss_res = np.sum((Δσ - Δσfit) ** 2)
            # total sum of squares
            ss_tot = np.sum((Δσ - np.mean(Δσ)) ** 2)    
            # r-squared
            r2 = 1 - (ss_res / ss_tot)
            #print(r2)
            if r2 > best_r2:
                best_r2 = r2
                best_popt = popt
                print(i)
            if np.mod(i,25) == 0:
                print(i)
                
        #result = annealing.curve_fit(WL_wiki2, H, Δσ, bounds=bounds)

        #best_popt = result.x # optimal fit parameters
        print(best_popt)
        Δσfit = WL_wiki(H, *best_popt)
        ax.plot(H, Δσfit *10**9, '-', ms=3, linewidth=1., color='black')
        
        Bϕ = best_popt[0]
        BSO = best_popt[1]
        Be = best_popt[2]
        #const = popt[3]
        
        # Bi = ħ/4e(li)^2   ->   li = sqrt(ħ/(4*e*Bi))
        Lϕ = np.sqrt(mp.ħ/(4*abs(mp.fundamental_charge_e*Bϕ)))
        LSO = np.sqrt(mp.ħ/(4*abs(mp.fundamental_charge_e*BSO)))
        Le = np.sqrt(mp.ħ/(4*abs(mp.fundamental_charge_e*Be)))
        print('Temperature %s K' % Temperature_K)
        print('Bϕ: %s T, Lϕ: %s m' % (Bϕ, Lϕ))
        print('BSO: %s T, LSO: %s m' % (BSO, LSO))
        print('Be: %s T, Le: %s m' % (Be, Le))
        #print('α: %s ' % const)
        
        print("r^2: %s" % best_r2)
    
    #ax.set_ylim((-1.2, .5))
    #ax.xaxis.set_major_locator(MultipleLocator(xmult))
    #ax.set_ylim((-5,105))

    mp.save_generic_svg(fig, device, savename)
    plt.show()
    plt.clf()

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
    
    #plot_σMCvsB_custom(fit=False)
    #plot_MR_vs_H_400K(size=2, fit=True)
    #plot_rho_vs_T(size=2, log=True)
    #plot_VH_vs_H_300K(size=2)
    #plot_VH_vs_H_400K(size=2)
    #plot_IDvsVDS(size=2)
    
    #plot_rho_vs_T_fits(size=2)
    
if __name__== "__main__":
  main()
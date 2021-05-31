# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:07:20 2019

@author: Justin
"""

import os
import sys
sys.path.append(os.path.join('..','..', 'Code'))
import material_plotter as mp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, LogLocator)

fileroot = r"JR190815_04"
JR190815_04_thickness = 110*10**-9 # meters (thickness)
JR190815_04_length = 12*10**-6 # meters
JR190815_04_width = 8*10**-6 # meters
JR190815_04_volt_spacing = 5*10**-6 # meters (center to center)             
                 
RTloop_filenames = [
    'JR190815_04_096_RvsVg_4.txt',
    'JR190815_04_097_RvsVg_10.txt',
    'JR190815_04_098_RvsVg_20.txt',
    'JR190815_04_099_RvsVg_30.txt',
    'JR190815_04_100_RvsVg_40.txt',
    'JR190815_04_101_RvsVg_50.txt',
    'JR190815_04_102_RvsVg_60.txt',
    'JR190815_04_103_RvsVg_70.txt',
    'JR190815_04_104_RvsVg_80.txt',
    'JR190815_04_105_RvsVg_90.txt',
    'JR190815_04_106_RvsVg_100.txt',
    'JR190815_04_108_RvsVg_150.txt',
    'JR190815_04_122_RvsVg_200.0K.txt',
    'JR190815_04_123_RvsVg_250.0K.txt',
    #'JR190815_04_124_RvsVg_270.0K.txt',
    #'JR190815_04_125_RvsVg_300.0K.txt',
    ]

rescan_RTloop_filenames = [
    'JR190815_04_096_RvsVg_4.txt',
    'JR190815_04_097_RvsVg_10.txt',
    'JR190815_04_098_RvsVg_20.txt',
    'JR190815_04_099_RvsVg_30.txt',
    'JR190815_04_100_RvsVg_40.txt',
    'JR190815_04_101_RvsVg_50.txt',
    'JR190815_04_102_RvsVg_60.txt',
    'JR190815_04_103_RvsVg_70.txt',
    'JR190815_04_104_RvsVg_80.txt',
    'JR190815_04_105_RvsVg_90.txt',
    'JR190815_04_106_RvsVg_100.txt',
    'JR190815_04_108_RvsVg_150.txt',
    'JR190815_04_122_RvsVg_200.0K.txt',
    'JR190815_04_123_RvsVg_250.0K.txt',
    'JR190815_04_124_RvsVg_270.0K.txt',
    'JR190815_04_125_RvsVg_300.0K.txt'
    ]

RTloop_2_4pt_filenames = [
    'JR190815_04_065_RvsVg_004_i.txt',
    'JR190815_04_063_RvsVg_050_i.txt',
    'JR190815_04_061_RvsVg_100_i.txt',
    'JR190815_04_059_RvsVg_150_i.txt',
    'JR190815_04_057_RvsVg_200_i.txt',
    'JR190815_04_055_RvsVg_250_i.txt',
    'JR190815_04_051_RvsVg_300_i.txt',
    'JR190815_04_053_RvsVg_270_i.txt',
    ]

Hall_filenames = [
    'JR190815_04_140_RvsB_75.00Vg_4.0K_hall.txt',
    'JR190815_04_142_RvsB_75.00Vg_50.0K_hall.txt',
    'JR190815_04_144_RvsB_75.00Vg_100.0K_hall.txt',
    'JR190815_04_146_RvsB_75.00Vg_200.0K_hall.txt',
    #'JR190815_04_148_RvsB_75.00Vg_300.0K_hall.txt',
    ]
      
def plot_R2pt_R4pt_DSvsVg():
    files = [mp.process_file(os.path.join(fileroot, x)) for x in RTloop_2_4pt_filenames]
    mp.plot_RDSvsVg_generic(fileroot, files, "_R2IDSvsVg(T)", R_ind=2)
    mp.plot_RDSvsVg_generic(fileroot, files, "_R4IDSvsVg(T)", R_ind=3)
    
    
def delta_R(file):
    gate_voltage = file['Gate_Voltage_V'].astype(int)
    resistance = file['Resistance_2_Ohms']
    uni_gate_voltage = np.unique(gate_voltage)
    
    deltaR = []
    for (gv) in uni_gate_voltage:
        indexes = np.argwhere(gate_voltage == gv)
        maxR = np.nanmax(resistance[indexes])
        minR = np.nanmin(resistance[indexes])
        if gv > 0:
            deltaR.append(maxR-minR)
        else:
            deltaR.append(np.NaN)
    
    return (uni_gate_voltage, deltaR)

def plot_deltaR():
    colors = ['#0000FF', '#FF0000', '#000000']
              
    filenames = ['JR190815_04_052_RvsVg_270_V.txt']
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=(3, 3*.90), dpi=300)
    ax = fig.add_subplot(111)
    

    ax.set_xlabel("$\it{V_{G}}$ (V)", fontname="Arial", fontsize=12)
    ax.set_ylabel('$\it{D-R}$ (Ohms)', fontname="Arial", fontsize=12)  # we already handled the x-label with ax1
    #ax.set_yscale('log')
    
    (gate_voltage, delta_Resistance) = delta_R(files[0])
    
    ax.plot(gate_voltage.astype(float), np.array(delta_Resistance)*(10.0**-6.0),
            '.-', ms=2, linewidth=1, color=colors[0])
    
    ax.set_xlim([-80,80])
    #ax.set_yticks([10**x for x in range(3,13)])
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax2 = ax.twinx()
    ax2.set_yscale(ax.get_yscale())
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(which='both', direction='in', labelright=False)
    ax2 = ax2.twiny()
    ax2.set_xscale(ax.get_xscale())
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(which='both', direction='in', labeltop=False)
    ax2.set_yticks(ax.get_yticks())
    ax2.minorticks_on()
    
    #plt.show()
    #ax.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')
    plt.tight_layout()
    #plt.axis('square')
    mp.save_generic_png(fig, fileroot, "_plot_deltaR_test")
    
def conducting_range(file):
    return [(gv, i) for (gv, i) in zip(file['Gate_Voltage_V'], file['Current_A']) if i > 10**-10]
    
def plot_onsetI():
    colors = ['#0000FF', '#FF0000', '#000000']
              
    filenames = rescan_RTloop_filenames
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=(3, 3*.90), dpi=300)
    ax = fig.add_subplot(111)
    

    ax.set_xlabel("$\it{T}$ (K)", fontname="Arial", fontsize=12)
    ax.set_ylabel('$\it{I}$ (μA)', fontname="Arial", fontsize=12)  # we already handled the x-label with ax1
    #ax.set_yscale('log')
    
    gv = []
    temp = []
    for file in files:
        gv.append(conducting_range(file)[0][0])
        temp.append(file['Temperature_K'][0])
    ax.plot(temp, gv, '.-', ms=2, linewidth=1, color=colors[0])
    
    ax.set_xlim([0,300])
    #ax.set_yticks([10**x for x in range(3,13)])
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax2 = ax.twinx()
    ax2.set_yscale(ax.get_yscale())
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(which='both', direction='in', labelright=False)
    ax2 = ax2.twiny()
    ax2.set_xscale(ax.get_xscale())
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(which='both', direction='in', labeltop=False)
    ax2.set_yticks(ax.get_yticks())
    ax2.minorticks_on()
    
    #plt.show()
    #ax.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')
    plt.tight_layout()
    #plt.axis('square')
    mp.save_generic_png(fig, fileroot, "_plot_onset_I")
    
def max_Vg(file):
    x = conducting_range(file)
    Vgs = [i[0] for i in x]
    Is = [i[1] for i in x]
    
    if len(Vgs) == 0:
        return (np.nan, np.nan, np.nan)
    
    maxVgs_index = np.argmax(Vgs)
    
    pre_Vgs = Vgs[:maxVgs_index+1]
    post_Vgs = np.flip(Vgs[maxVgs_index:])
    pre_Is = Is[:maxVgs_index+1]
    post_Is = np.flip(Is[maxVgs_index:])
    
    maxVg = (0, 0, 0) 
    
    i = j = 0
    if pre_Is[0] < post_Is[0]:
        i+=1
    else:
        j+=1
    while(i+1 < len(pre_Vgs) and j+1 < len(post_Vgs)):
        # four points, two on left, two on right
        y0 = pre_Is[i]; x0 = pre_Vgs[i]
        y1 = pre_Is[i+1]; x1 = pre_Vgs[i+1]
        y2 = post_Is[j]; x2 = post_Vgs[j]
        y3 = post_Is[j+1]; x3 = post_Vgs[j+1]
        
        if (y2 < y0 and y0 < y3):
            m = (y3 -y2)/(x3-x2)
            b = y2 - m*x2
            x = (y0 - b)/m
            dVg = x - x0
            Vg0 = x0
            Is0 = y0
            i+=1
        elif (y0 < y2 and y2 < y1):
            m = (y1 - y0)/(x1 - x0)
            b = y0 - m*x0
            x = (y2 - b)/m
            dVg = x2 - x
            Vg0 = x
            Is0 = y2
            j+=1
        elif (y2 > y1):
            i+=1
            continue
        elif (y0 > y3):
            j+=1
            continue
        else:
            print("bahhhhhhhhhhhhhhhhhhhhhhhhh")
            continue
        
        if (maxVg[0] < dVg):
            maxVg = (dVg, Vg0, Is0)
            
    return maxVg
    
    
def plot_delta_Vgmax():
    colors = ['#0000FF', '#FF0000', '#000000']
              
    filenames = rescan_RTloop_filenames
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=(3, 3*.90), dpi=300)
    ax = fig.add_subplot(111)
    
    

    ax.set_xlabel("$\it{T}$ (K)", fontname="Arial", fontsize=12)
    ax.set_ylabel('$\it{ΔV_{G}^{max}}$ (V)', fontname="Arial", fontsize=12, color=colors[0])  # we already handled the x-label with ax1
    #ax.set_yscale('log')
    
    dgv = []
    Imax = []
    temp = []
    for file in files:
        x = max_Vg(file)
        dgv.append(x[0])
        Imax.append(x[2]*(10**9))
        temp.append(file['Temperature_K'][0])
    ax.plot(temp, dgv, '.-', ms=2, linewidth=1, color=colors[0])
    
    ax.set_xlim([0, 300])
    #ax.set_yticks([10**x for x in range(3,13)])
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax2 = ax.twinx()
    #ax2.set_yscale(ax.get_yscale())
    #ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel('$\it{@I_{D}}$ (nA)', fontname="Arial", fontsize=12, color=colors[1])
    ax2.tick_params(which='both', direction='in', labelright=True)
    ax2 = ax2.twiny()
    ax2.set_xscale(ax.get_xscale())
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(which='both', direction='in', labeltop=False)
    #ax2.set_yticks(ax.get_yticks())
    ax2.minorticks_on()
      
    ax2.plot(temp, Imax, '.-', ms=2, linewidth=1, color=colors[1])
    
    #plt.show()
    #ax.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')
    plt.tight_layout()
    #plt.axis('square')
    mp.save_generic_png(fig, fileroot, "_Vg_max")

def plot_contact_generic(filename, colors, savename):
    fig = plt.figure(figsize=(2, 2), dpi=300)
    
    files = [mp.process_file(os.path.join(fileroot, filename))]
    
    occ0 = mp.first_occurance_1D(files[0]['Gate_Voltage_V'], 0, tol=0.2, starting_index=0)
    occ1 = mp.first_occurance_1D(files[0]['Gate_Voltage_V'], 75.0, tol=0.2, starting_index=occ0+1)
    file = files[0][occ0:occ0+occ1+2]
    files = [file]
    
    ax = mp.pretty_plot_single(fig, labels=['$\it{V_{G}}$ (V)', '$\it{R}$ (Ω)'],
                             yscale='log', fontsize=10, labelsize=10)
    
    for file in files:
        ax.plot(file['Gate_Voltage_V'], np.abs(file['Resistance_1_Ohms']), '.-', ms=2, linewidth=1, color=colors[0])
        ax.plot(file['Gate_Voltage_V'], np.abs(file['Resistance_3_Ohms']), '.-', ms=2, linewidth=1, color=colors[1])
   
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlim((-3,83))
    
    mp.save_generic_svg(fig, fileroot, savename)
    
def plot_contact():
    filenames = ['JR190815_04_136_RvsVg_300.0K_contact.txt',#0.0, V1+ to V1-; S to V1+ ; c0, c1
                 #'JR190815_04_137_RvsVg_100.0K_contact.txt',#1.0, V1+ to V1-; S to V1+
                 #'JR190815_04_135_RvsVg_270.0K_contact.txt',#2.0, V1+ to V1-; S to V1+
                 #'JR190815_04_134_RvsVg_270.0K_contact.txt',#3.0, S to V1+; V1+ to V1- ; c1, c0
                 #'JR190815_04_131_RvsVg_200.0K_contact.txt',#4.0, S to V1+; V1+ to V1-
                 #'JR190815_04_130_RvsVg_100.0K_contact.txt',#5.0, S to V1+; V1+ to V1-  
                 #'JR190815_04_149_RvsVg_300.0K_contact2.txt',#6.1, V1+ to V1-; V1- to D ; c0, c2
                 #'JR190815_04_150_RvsVg_200.0K_contact2.txt',#7.1, V1+ to V1-; V1- to D 
                 #'JR190815_04_151_RvsVg_4.0K_contact2.txt',#8.1, V1+ to V1-; V1- to D 
                 #'JR190815_04_153_RvsVg_4.0K_contact3.txt',#9.2, V2+ to V2-; V2- to D ; c3, c2
                 #'JR190815_04_154_RvsVg_100.0K_contact3.txt',#10.2, V2+ to V2-; V2- to D 
                 #'JR190815_04_155_RvsVg_200.0K_contact3.txt',#11.2, V2+ to V2-; V2- to D 
                 #'JR190815_04_156_RvsVg_300.0K_contact3.txt',#12.2, V2+ to V2-; V2- to D 
                 #'JR190815_04_157_RvsVg_300.0K_contact4.txt'#13.3, S to V2+; V2+ to V2- ; c4, c3
                ]
    colors = mp.colors_set1
    colors = [[colors[0], colors[1]],#0
              [colors[0], colors[1]],#1
              [colors[0], colors[1]],#2
              [colors[1], colors[0]],#3
              [colors[1], colors[0]],#4
              [colors[1], colors[0]],#5
              [colors[0], colors[2]],#6
              [colors[0], colors[2]],#7
              [colors[0], colors[2]],#8
              [colors[3], colors[2]],#9
              [colors[3], colors[2]],#10
              [colors[3], colors[2]],#11
              [colors[3], colors[2]],#12
              [colors[4], colors[3]],
            ]
    savename = [x.replace('.txt','') for x in filenames]
    print(filenames)
    for (i, j, k) in zip(filenames, colors, savename):
        plot_contact_generic(i, j, k)


def plot_hall_V_generic(H_data, V_data, savename, polyfit=None):
    (vscale, vlabel) = mp.m_order(V_data)
    fig = plt.figure(figsize=(2, 2), dpi=300)
    
    (ax, ax2, ax3) = mp.pretty_plot(fig, labels=['$\it{H}$ (T)', '$\it{V_{H}}$ (%sV)' % vlabel],
                             yscale=['linear','linear'])
    ax.plot(H_data, V_data*vscale, '.-', ms=3, linewidth=1.5, color=mp.colors_set1[0])
    
    if polyfit is not None:
        ax.plot(H_data, polyfit(H_data)*vscale, '-', ms=3, linewidth=1.5, color=mp.colors_set1[1])
    
    ax.set_xlim((-11, 11))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    ax.minorticks_on()
    ax2.minorticks_on()
    
    mp.save_generic_svg(fig, fileroot, savename)
    

def process_hall_data():
    Temperatures = []
    offset = [0, 0, 1, 0]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in Hall_filenames]
    
    #RTloop_2_4pt_filenames
    #rescan_RTloop_filenames
    #RTloop_filenames
    (cs_Currents_left, cs_Voltages_left, _, cs_Temperatures) = \
        mp.get_cross_section(fileroot, RTloop_2_4pt_filenames, [75.], 1)
    (cs_Currents_right, cs_Voltages_right, _, cs_Temperatures) = \
        mp.get_cross_section(fileroot, RTloop_2_4pt_filenames, [75.], 2)
    
    n2Ds = [] # 2D hall density
    r_squared = [] # error
    fits = [] 
    μH = [] # mobility
    for (file, off) in zip(files, offset):
        current = file['Current_A'][0]
        T = file['Temperature_K'][0]
        Temperatures.append(T)
        
        # hall data for line
        B_data = file['Magnetic_Field_T'][off:]
        V_Hdata = file['Voltage_1_V'][off:]
        VDS_data = file['Voltage_3_V'][off:]
        
        # fit V_Hall to a line
        (pcoefs, residuals, rank, singular_values, rcond) = \
            np.polyfit(B_data, V_Hdata, 1, full = True)
        
        pfit = np.poly1d(pcoefs)
        fits.append(pfit)
        
        # error in fit
        residuals = V_Hdata - pfit(B_data)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((V_Hdata-np.mean(V_Hdata))**2)
        r_squared.append(1 - (ss_res / ss_tot))
        
        # pick a point on the line
        B_point = 1.0 # Tesla
        V_hall0T = pfit(0)
        V_hall_T = pfit(B_point)
        
        # Hall resistance V(B)/(B*I), remove 0T offset
        #RH2D = ((V_hall_T-V_hall0T)/current)/B_point
        
        # n2D = B/(Rxy*e) = 1/RH2D*e 
        n2D = -B_point*current/((V_hall_T-V_hall0T)*abs(mp.fundamental_charge_e))
        n2Ds.append(n2D)
        
        # R from 2pt resistance in actual measurement
        ind = mp.first_occurance_1D(B_data, 0, tol=0.01, starting_index=0)
        R_2pt = VDS_data[ind]/current ##
        σs = JR190815_04_length / (R_2pt * JR190815_04_width)
        
        # pull R from seperate 4pt data instead
        occ0 = mp.first_occurance_1D(cs_Temperatures, T, tol=.2, starting_index=0)
        R_left = cs_Voltages_left[0][occ0]/cs_Currents_left[0][occ0]
        R_right = cs_Voltages_right[0][occ0]/cs_Currents_right[0][occ0]
        #print("R_left: %s Ω and R_right %s Ω" % (round(R_left), round(R_right)))
        Rxx_4pt = (R_left+R_right)/2
        
        #σs = l/(Rxx*w)
        σs = JR190815_04_volt_spacing / (Rxx_4pt * JR190815_04_width)
        
        #print("R_DS: %s Ω and R_loop: %s Ω" % (round(R_2pt), round(Rxx_4pt)))
        
        # μ = σs/(e*n2D)
        μ = σs/(abs(mp.fundamental_charge_e)*n2D)
        μH.append(μ)
        
        #plot_hall_V_generic(xdata, ydata, '_RvsH_' + str(temp) + 'K_R2_' + str(r_squared), polyfit=pfit)
        print("%s K: μH: %s cm^2/Vs" % (round(T,1), np.multiply(μ,100*100)))
    
    print("Fit R^2: %s" % r_squared)
    #print("μH: %s cm^2/Vs" % np.multiply(μH,100*100))
    
    return (np.array(Temperatures), n2Ds, fits, r_squared, np.array(μH))
 
def plot_n_hall_3D():
    (Temperatures, n2D, fits, r_squared, μH) = process_hall_data()
    n3D = np.divide(n2D,JR190815_04_thickness)/(100.0**3.0)
        
    fig = plt.figure(figsize=(2, 2), dpi=300)
    
    ax = mp.pretty_plot_single(fig, labels=['$\it{T}$ (K)', '$\it{n}$ ($cm^{-3}$)'],
                             yscale='log', fontsize=10)
    
    ax.plot(Temperatures, n3D, '.-', ms=3, linewidth=1.5)
   
        
    ax.set_xlim((None, 215))
    ax.set_ylim( (.9*10.0**17.0,1.3*10.0**18.0) )
    ax.xaxis.set_major_locator(MultipleLocator(100))

    
    mp.save_generic_svg(fig, fileroot, "_Hall_n3D")
    
def plot_n_hall_2D(fontsize=10, labelsize=10):
    (Temperatures, n2D, fits, r_squared, μH) = process_hall_data()
    
    # setup
    log = True
    fig = plt.figure(figsize=(2, 2), dpi=300)
    y_pow = 1
    y_label = ''
    m2_to_cm2 = (1.0*10**-4)
    
    if not log:
        ymax = [np.nanmax(n2D)*m2_to_cm2]
        (y_pow, y_label) = mp.m_order(ymax)
    
    ax = mp.pretty_plot_single(fig, labels=['$\it{T}$ (K)', '$\it{n}$ ($%s cm^{-2}$)' % (y_label)],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)
    
    
    ax.plot(Temperatures, np.multiply(n2D,m2_to_cm2*y_pow), '.-', ms=3, linewidth=1.5) 
    
    # axis
    ax.set_xlim((None, 215))
    ax.set_ylim( (1*10.0**12,1.5*10.0**13.0) )
    ax.xaxis.set_major_locator(MultipleLocator(100))

    mp.save_generic_both(fig, fileroot, "_Hall_n2D")
    plt.show()
    plt.clf()
       

def plot_hall_inset(log = False, fontsize=10, labelsize=10):
    offset = [0, 0, 1, 0]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in Hall_filenames]
    
    colors=mp.colors_set1[[4, 3, 2, 0]]
    
    # scale
    
    ymax = [10.] if log else [np.nanmax(file['Resistance_1_Ohms']) for file in files]
    (scale_pow, scale_label) = mp.m_order(ymax)
    
    fig = plt.figure(figsize=(1., 1.), dpi=300)
    ax = mp.pretty_plot_single(fig, labels=['$\it{B}$ (T)', '$\it{R_{xy}}$ (%sΩ)' % scale_label],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)
    
    for i, (file, off) in enumerate(zip(files, offset)):
        xdata = file['Magnetic_Field_T'][off:]
        ydata = file['Resistance_1_Ohms'][off:]
        
        (pcoefs, residuals, rank, singular_values, rcond) = \
            np.polyfit(xdata, ydata, 1, full = True)
        
        pfit = np.poly1d(pcoefs)
        
        #residuals = ydata - pfit(xdata)
        #ss_res = np.sum(residuals**2)
        #ss_tot = np.sum((ydata-np.mean(ydata))**2)
        
        #RH = thickness*((pfit(B_point)-pfit(-B_point))/2)/B_point
        
        start = mp.first_occurance_1D(xdata, 0, tol=0.01)
        
        ax.plot(xdata[start:], -(pfit(xdata[start:])-pfit(0))*scale_pow, '-', ms=3, linewidth=1, color=colors[i])
        ax.plot(xdata[start:], -(ydata[start:]-pfit(0))*scale_pow, '.', ms=3, linewidth=1.5, color=colors[i])
    
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    
    ax.minorticks_on()
    
    mp.save_generic_svg(fig, fileroot, "_Hall_stacked")

def plot_IV_generic(files, savename, colors, log=False, invertaxes=False):    
    fig = plt.figure(figsize=(2, 2), dpi=300)
    pre = '-' if invertaxes else ''

    ax= mp.pretty_plot_single(fig, labels=["$\it{%sV_{DS}}$ (V)" % pre, '$\it{%sI_{D}}$ (μA)' % pre],#μ
                             yscale=('log' if log else 'linear'), fontsize=10, labelsize=8, labelpad=[0,0])
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Voltage_1_V'], (file['Current_A'])*(10**6), '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if invertaxes:
        ax.set_xlim(xlim[1], xlim[0]-.5)
        ax.set_ylim(ylim[1], ylim[0]*1.1)
    else:
        ax.set_xlim(xlim[0], xlim[1]+.5)
        ax.set_ylim(ylim[0], ylim[1]*1.1)

    mp.save_generic_svg(fig, fileroot, savename)

def plot_300K_IDvsVDS(figsize=1.5, log=False):
    colors = mp.colors_set1
    colors = [colors[0], colors[4], colors[3], colors[6], colors[2], colors[8], colors[1]]
    
    start = 10
    savename = "_JR190815_04_300K_"
    
    xadj = 1
    filenames = [('JR190815_04_' + str(i).zfill(3) + '_IvsV_increase.txt') for i in range(start, start+7)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + 'positive_increasing', colors,\
                              size=figsize, xadj=xadj, log=log)
    
    filenames = [('JR190815_04_' + str(i).zfill(3) + '_IvsV_increase.txt') for i in range(start+7, start+14)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + 'positive_decreasing', colors[::-1],\
                              size=figsize, xadj=xadj, log=log)
    
    filenames = [('JR190815_04_' + str(i).zfill(3) + '_IvsV_decrease.txt') for i in range(start+14, start+21)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + 'negative_increasing', colors,\
                              size=figsize, invertaxes=True, xadj=xadj, log=log)
    
    filenames = [('JR190815_04_' + str(i).zfill(3) + '_IvsV_decrease.txt') for i in range(start+21, start+28)]
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
    mp.plot_IDvsVDS_generic(fileroot, files, savename + 'negative_decreasing', colors[::-1],\
                              size=figsize, invertaxes=True, xadj=xadj, log=log)
    
def main(): #sample D
    show_all = False
    # Plot ID vs VG loops
    if False or show_all:
        mp.plot_IDvsVg_each(fileroot, RTloop_filenames, '_JR190815_04', log=True, size=2, majorx=40,
                          ylim=(None,None), fontsize=10, labelsize=10)
    
    # -- Cross section of loop data --
    if False or show_all:
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR190815_04_RDS", increments=[0,25,50,75],\
                                      figsize=2, log=True, xlim=(0, 320),ylim=(None, None))
    
    # -- 300K ID vs VDS curves
    if False or show_all:
        plot_300K_IDvsVDS(figsize=2, log=False)
        plot_300K_IDvsVDS(figsize=2, log=True)

    # hall carrier density
    if False or show_all:
        plot_n_hall_2D()
        plot_hall_inset()
        plot_n_hall_3D()
    
    # contact resistance
    if False or show_all:
        plot_contact()
    
    # unused but saving
    if False:  
        plot_deltaR()
        plot_onsetI()
        plot_R2pt_R4pt_DSvsVg()
        plot_delta_Vgmax()
        plot_n_hall_3D()
    
    # -- carrier mobility μ
    if False or show_all:
        mp.plot_mobility_μ_cross_section(fileroot, RTloop_filenames, "_JR190815_04", JR190815_04_length, JR190815_04_width, figsize=1.5, ylim=(None, None),\
                                           log=False, increments=[25, 50, 75], colororder=[3,2,1])
        
    # min subthreshold slope
    if True or show_all:
        mp.plot_maxSS_vs_T(fileroot, RTloop_filenames, '_minSSvsT', Npoints=5, Icutoff=10*10**-11)
        
    #files = [mp.process_file(os.path.join(fileroot, x)) for x in ['JR190815_04_125_RvsVg_300.0K.txt']]
    #print(mp.width_Vg(files[0],10**-8))
    
    
if __name__== "__main__":
  main()

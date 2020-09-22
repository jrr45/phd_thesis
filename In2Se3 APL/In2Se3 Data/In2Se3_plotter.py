import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Jpython_plotter as jpp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, LogLocator)

fundamental_charge_e = -1.602176634 * (10**-19) # SI: C
ħ = 1.0545718 * 10**-34 #SI m^2 kg / s

# cross sections of gating loops
def get_cross_section(fileroot, filenames, increments, V_index):
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    indexes = []
    for inc in increments:
        indexes.append([jpp.first_occurance_1D(file['Gate_Voltage_V'], inc, starting_index=5)+5 for file in files])
    
    Currents = []
    Voltages = []
    Gate_Voltages = []
    for index in indexes:    
        Currents.append([file['Current_A'][i]                      for (file, i) in zip(files, index)])
        Voltages.append([file['Voltage_' + str(V_index) + '_V'][i] for (file, i) in zip(files, index)])
        Gate_Voltages.append([file['Gate_Voltage_V'][i]            for (file, i) in zip(files, index)])

    Temperatures = np.array([(file['Temperature_K'][0]) for file in files])
    
    return (np.array(Currents), np.array(Voltages), np.array(Gate_Voltages), Temperatures)

def plot_loopI_cross_section(fileroot, filenames, savename, increments=[0,25,50,75], \
                             figsize=2, xlim=(None, 330), ylim=(None,None), log=True, \
                             fontsize=10, labelsize=10, colororder=[0,3,2,1,4,5]):
    colors = jpp.colors_set1[colororder]
              
    (Currents, Voltages, GateVoltages, Temperatures) = get_cross_section(fileroot, filenames, increments, 1)
    ymax = [10.] if log else np.nanmax([np.nanmax(y) for y in Currents]) 
    (scale_pow, scale_label) = jpp.m_order(ymax)
    
    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{I_{D}}$ (%sA)' % scale_label],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)

    for (color, Is) in zip (colors, Currents):
        ax.plot(Temperatures, Is*scale_pow, '.-', ms=3, linewidth=1.5, color=color)
    
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(xlim)
    
    scalename = "_log" if log else "_linear"
    print(savename+  "_loop_I-cross" +scalename)
    jpp.save_generic_svg(fig, fileroot, savename + "_loop_I-cross" + scalename)
    plt.show()
    plt.clf()

def plot_loopR_cross_section(fileroot, filenames, savename, increments=[0,25,50,75], \
                             figsize=2, xlim=(None, 330), xinc=100, ylim=(None,None), log=True, \
                             fontsize=10, labelsize=10, colororder=[0,3,2,1,4,5]):
    colors = jpp.colors_set1[colororder]
    
    (Currents, Voltages, GateVoltages, Temperatures) = get_cross_section(fileroot, filenames, increments, 1)
    Resistances = []
    for (Vs, Is) in zip(Voltages, Currents):    
        Resistances.append(np.array([v/i for (v,i) in zip(Vs,Is)]))

    ymax = [10.] if log else np.nanmax([np.nanmax(R) for R in Resistances]) 
    (scale_pow, scale_label) = jpp.m_order(ymax)

    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{R_{DS}}$ (%sΩ)' % scale_label],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)

    for (color, Rs) in zip (colors, Resistances):
        #real filter
        ind = np.where(np.logical_and(np.isfinite(Rs), Rs > 0))
        Rs = Rs[ind]
        T = Temperatures[ind]
        ax.plot(T, abs(Rs)*scale_pow, '.-', ms=3, linewidth=1.5, color=color)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(xinc))
    
    scalename = "_log" if log else "_linear"
    print(savename+  "_loop_R-cross" +scalename)
    jpp.save_generic_svg(fig, fileroot, savename + "_loop_R-cross" + scalename)
    plt.show()
    plt.clf()
    
    
def plot_mobility_μ_cross_section(fileroot, filenames, savename, length, width, increments=[0,25,50,75], \
                             figsize=3, xlim=(None, 330), ylim=(None,None), log=True, fontsize=10, labelsize=10, colororder=[0,3,2,1,4,5]):
    colors = jpp.colors_set1[colororder]
    
    (Currents, Voltages, GateVoltages, Temperatures) = get_cross_section(fileroot, filenames, increments, 1)
    
    mobilities = []
    for (Vs, Is, V_G) in zip(Voltages, Currents, increments):    
        Rs = np.array([v/i for (v,i) in zip(Vs,Is)])
        n = wafer_carrier_density_n(V_G)
        σs = sheet_conductivity(Rs, length, width)
        μs = carrier_mobility_μ(σs, n)
        mobilities.append(μs)
        
    unit_scale = (100.*100.) #m^2 in cm^2
    mobilities = np.array(mobilities) * unit_scale
    
    ymax = [10.] if log else np.nanmax([np.nanmax(y) for y in mobilities]) 
    (scale_pow, scale_label) = jpp.m_order(ymax)
    
    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{μ}$ (%s $cm^2/V\\cdot s$)' % scale_label],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)

    for (color, μs, VG) in zip (colors, mobilities, increments):
        #real filter
        ind = np.where(np.logical_and(np.isfinite(μs), μs > 0))
        μs = μs[ind]
        T = Temperatures[ind]
        print("VG: %s V, μ:%s" %(VG, μs))
        ax.plot(T, np.multiply(μs, scale_pow), '.-', ms=3, linewidth=1.5, color=color)

    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_ylim(ylim)

    scalename = "_log" if log else "_linear"
    print(savename+  "_loop_μ-cross" +scalename)
    jpp.save_generic_svg(fig, fileroot, savename + "_loop_μ-cross" + scalename)
    plt.show()
    plt.clf()

# generic functions
def plot_IDvsVg_each(fileroot, filenames, savename, log=False, size=1.5, majorx=25, ylim=(None,None), fontsize=10, labelsize=10):
   
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    for file in files:
        Iname = '_IDvsVg_' + str(round(file['Temperature_K'][0],1)).zfill(5) + 'K'
        (file0, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
                
        #highpass filter
        ind = np.where(file0['Current_A'] > 2*10**-12)
        file0 = file0[ind]
        
        plot_IDvsVg_generic(fileroot, [file0], savename + Iname, [jpp.colors_set1[1]], log=log, \
                              size=size, majorx=majorx, ylim=ylim, fontsize=fontsize, labelsize=labelsize)
        
def plot_IDvsVg_generic(fileroot, files, savename, colors, log=False, size=1.5, majorx=40, ylim=(None,None),\
                        fontsize=10, labelsize=10):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    scale_pow = 1
    scale_label = ''
    
    if not log:
        Imax = [np.nanmax(file['Current_A']) for file in files]
        (scale_pow, scale_label) = jpp.m_order(Imax)
    
    ax = jpp.pretty_plot_single(fig, labels=["$\it{V_{G}}$ (V)", '$\it{I_{D}}$ (%sA)' % (scale_label)],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Gate_Voltage_V'], np.abs(file['Current_A']) if log else (file['Current_A'])*scale_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(majorx))
    ax.set_ylim(ylim)
    
    scalename = "_log" if log else "_linear"
    print(savename+'_IDvsVg_'+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+'_IDvsVG'+scalename)
    plt.show()
    plt.clf()

def plot_IDvsVg_subleak_generic(fileroot, files, savename, colors, log=False, size=1.5, majorx=25):    
    for file in files:
        file['Current_A'] = file['Current_A']-file['Gate_Leak_Current_A']
        
    plot_IDvsVg_generic(fileroot, files, savename, colors, log=log, size=size, majorx=majorx)
    
def plot_IleakvsVg_generic(fileroot, files, savename, colors, log=False, adjust=True):    
    fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
    scale_pow = 1
    scale_label = ''
    
    if not log:
        Imax = [np.nanmax(file['Gate_Leak_Current_A']) for file in files]
        (scale_pow, scale_label) = jpp.m_order(Imax)
    
    ax = jpp.pretty_plot_single(fig, labels=["$\it{V_{G}}$ (V)", '$\it{I_{leak}}$ (%sA)' % (scale_label)],
                             yscale='log' if log else 'linear', fontsize=10)
    
    ax.minorticks_on()
    
    occ0 = jpp.first_occurance_1D(files[0]['Gate_Voltage_V'], -75.0, tol=0.2, starting_index=0)
    occ1 = jpp.first_occurance_1D(files[0]['Gate_Voltage_V'], -75.0, tol=0.2, starting_index=occ0+1)
    files = [files[0][occ0:occ0+occ1+2]]
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Gate_Voltage_V'], np.abs(file['Gate_Leak_Current_A']) if log else (file['Gate_Leak_Current_A'])*scale_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(40))

    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()

def plot_RDSvsVg_generic(fileroot, files, savename, colors=jpp.colors_set1, R_ind=1, log=False, size=1.5, majorx=25, ylim=(None,None)):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    R_pow = 1
    R_label = ''
    R_col = 'Resistance_' + str(R_ind) + '_Ohms'
    
    if not log:
        Rmax = [np.nanmax(file[R_col]) for file in files]
        (R_pow, R_label) = jpp.m_order(Rmax)
    
    ax = jpp.pretty_plot_single(fig, labels=["$\it{V_{G}}$ (V)", '$\it{R_{DS}}$ (%sΩ)' % (R_label)],
                             yscale=('log' if log else 'linear'))
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Gate_Voltage_V'], np.abs(file[R_col]) if log else (file[R_col])*R_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(majorx))
    ax.set_ylim(ylim)
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()

def plot_IDvsVDS_generic(fileroot, files, savename, colors, log=False, invertaxes=False, figsize=2, \
                         xadj=0, x_mult=5, fontsize=10, labelsize=10, ylim=None, ms=2, linewidth=1.5, \
                         linedot='.-'):    
    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    pre = '-' if invertaxes else ''
    
    ymax = [10.] if log else [np.nanmax(np.abs(file['Current_A'])) for file in files]
    (scale_pow, scale_label) = jpp.m_order(ymax)
        
    ax= jpp.pretty_plot_single(fig, labels=["$\it{%sV_{DS}}$ (V)" % pre,\
                                            '$\it{%sI_{D}}$ (%sA)' % (pre, scale_label)],
                               yscale=('log' if log else 'linear'),
                               fontsize=fontsize, labelsize=labelsize, labelpad=[0,0])
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Voltage_1_V'], (np.abs(file['Current_A']) if log else (file['Current_A'])*scale_pow),\
                linedot, ms=ms, linewidth=linewidth, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(x_mult))
    #if not log:
    #    ax.yaxis.set_major_locator(MultipleLocator(.2))
    
    xlim = ax.get_xlim()
    auto_ylim = ax.get_ylim()
    if invertaxes:
        ax.set_xlim(xlim[1], xlim[0]-xadj)
        if ylim is not None:
            ax.set_ylim(ylim)
        elif not log:
            ax.set_ylim(auto_ylim[1]+.05, auto_ylim[0]*1.1)
    else:
        ax.set_xlim(xlim[0], xlim[1]+xadj)
        if ylim is not None:
            ax.set_ylim(ylim)
        elif not log:
            ax.set_ylim(auto_ylim[0]-.05, auto_ylim[1]*1.1)
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()
    
# 300K IV plots
def plot_IDvVDS_gating_generic(fileroot, sample, end_name, startindex, incindex, savename, \
                               figsize=2, xadj=1, log=False):
    colors = jpp.colors_set1[[0,4,3,6,2,8,1]]
    
    start = startindex 
    inc = incindex
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start, start+1*inc)]
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'positive_increasing', colors,\
                         figsize=figsize, xadj=xadj, log=log)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+1*inc, start+2*inc)]
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'positive_decreasing', colors[::-1],\
                         figsize=figsize, xadj=xadj, log=log)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+2*inc, start+3*inc)]
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'negative_increasing', colors,\
                         figsize=figsize, invertaxes=True, xadj=xadj, log=log)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+3*inc, start+4*inc)]
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'negative_decreasing', colors[::-1],\
                         figsize=figsize, invertaxes=True, xadj=xadj, log=log)
    
def plot_IDvsB_generic(fileroot, filenames, savename, colors, log=False, symm=False, size=2):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    scale_pow = 1
    scale_label = ''
    ystring = 'Current_A'
    xstring = 'Magnetic_Field_T'
    
    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]

    ymax = [10.] if log else [np.nanmax(np.abs(file[ystring])) for file in files]
    (scale_pow, scale_label) = jpp.m_order(ymax)
    
    ax = jpp.pretty_plot_single(fig, labels=["$\it{B}$ (T)", '$\it{I_{D}}$ (%sA)' % (scale_label)],
                             yscale=('log' if log else 'linear'))
    
    for (file, color, color2) in zip(files, colors[::2], colors[1::2]):
        if symm:
            curr = file[ystring]
            curr = (curr+curr[::-1])/2
            ax.plot(file[xstring], np.abs(curr) if log else (curr)*scale_pow,
                '.-', ms=3, linewidth=1.5, color=color2)
        else:
             ax.plot(file[xstring], np.abs(file[ystring]) if log else (file[ystring])*scale_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()

def width_Vg(file, current):
    #relevant ranges
    (file1, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', -75., 75., .01, starting_index=0)
    (file2, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', 75., -75., .01, starting_index=end_ind)
    
    Vgs1 = file1['Gate_Voltage_V']
    Is1 = file1['Current_A']
    Vgs2 = file2['Gate_Voltage_V']
    Is2 = file2['Current_A']
    
    if len(Vgs1) == 0 or len(Vgs2) == 0:
        return np.nan
    
    # find the points just above and below the given current
    Vgs1_index1 = np.argmin(np.abs(Is1 - current))
    if (Is1[Vgs1_index1] - current < 0 and Is1[Vgs1_index1+1] - current > 0) or \
        (Is1[Vgs1_index1] - current > 0 and Is1[Vgs1_index1+1] - current < 0):
        Vgs1_index2 = Vgs1_index1+1
    else:
        Vgs1_index2 = Vgs1_index1-1
    
    Vgs2_index1 = np.argmin(np.abs(Is2 - current))
    if (Is2[Vgs2_index1] - current < 0 and Is2[Vgs2_index1+1] - current > 0) or \
        (Is2[Vgs2_index1] - current > 0 and Is2[Vgs2_index1+1] - current < 0):
        Vgs2_index2 = Vgs2_index1+1
    else:
        Vgs2_index2 = Vgs2_index1-1

    # fit the line formula, y = mx + b 
    m1 = (Is1[Vgs1_index2] - Is1[Vgs1_index1]) / (Vgs1[Vgs1_index2] - Vgs1[Vgs1_index1])
    b1 = Is1[Vgs1_index1] - m1*Vgs1[Vgs1_index1]
    
    m2 = (Is2[Vgs2_index2] - Is2[Vgs2_index1]) / (Vgs2[Vgs2_index2] - Vgs2[Vgs2_index1])
    b2 = Is2[Vgs2_index1] - m2*Vgs2[Vgs2_index1]
    
    # calc VG at current value
    VG1 = (current - b1)/m1
    VG2 = (current - b2)/m2
    
    return np.abs(VG2 - VG1)

def plot_ΔVGvT(fileroot, filenames, current, size=2, log=False):
    savename = '_DVGvT'
    size = 2
    colors = jpp.colors_set1

    files = [jpp.process_file(os.path.join(fileroot, x)) for x in filenames]
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{ΔV_{G}}$ (V)'],
                             yscale=('log' if log else 'linear'))
    
    DVG = []
    T = []
    for file in files:
        T.append(file['Temperature_K'][0])
        dVg = width_Vg(file, current)
        DVG.append(dVg)
        
    ax.plot(T, DVG, '.-', ms=3, linewidth=1.5, color=colors[0])
    
    ax.xaxis.set_major_locator(MultipleLocator(100))
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()
    
def plot_IDSvsTime_generic(fileroot, files, savename, colors=jpp.colors_set1, log=False, size=2, majorx=1800, ylim=(None,None)):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    y_pow = 1
    y_label = ''
    ystring = 'Current_A'
    
    ymax = [10.] if log else [np.nanmax(np.abs(file[ystring])) for file in files]
    (scale_pow, scale_label) = jpp.m_order(ymax)
    
    ax = jpp.pretty_plot_single(fig, labels=["$\it{t_{s}}$ (V)", '$\it{I_{D}}$ (%sA)' % (y_label)],
                             yscale=('log' if log else 'linear'))
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Time_s'], np.abs(file[ystring]) if log else (file[ystring])*y_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(majorx))
    ax.set_ylim(ylim)
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    jpp.save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()
    
def slice_data(file, column, start, end, tol, nth_start=1, nth_finish=1, starting_index=0):
    if start is None:
        occ0 = 0
    else:
        occ0 = jpp.nth_occurance_1D(file[column], start, nth_start, tol=tol, starting_index=starting_index)
    if end is None:
        occ1 = np.size(file) - occ0 - 1
    else:
        occ1 = jpp.nth_occurance_1D(file[column], end, nth_finish, tol=tol, starting_index=occ0+1)
    file2 = np.copy(file[occ0:occ0+occ1+2])
    
    return (file2, occ0+starting_index, occ1+occ0+starting_index)

def slice_data_each(files, column, start, end, tol, nth_start=1, nth_finish=1, starting_index=0):
    for i, file in enumerate(files):
        (files[i],_,_) = slice_data(file, column, start, end, tol, nth_start=nth_start,
                                    nth_finish=nth_finish, starting_index=starting_index)
    return files

def sheet_conductivity(resistance_2T, length, width):
    # σs = L/(R*W)
    return length/np.multiply(resistance_2T, width)

def sheet_conductivity_4pt(resistance_2T, length, width):
    # σs = L/(R*W) FIXME
    return 0*length/np.multiply(resistance_2T, width)

def sheet_resistance(resistance_2T, length, width):
    # R□ = Rxx*W/(L)
    return np.multiply(resistance_2T, width)/length

def carrier_mobility_μ(sheet_conductivity_σs, carrier_density_n):
    # σs =  μen
    μ = sheet_conductivity_σs/(carrier_density_n * abs(fundamental_charge_e))
    return μ

def wafer_carrier_density_n(gate_voltage):
    #n2D = εr*ε0*V/(d*q)
    thickness = 300*(10**-9) #300nm
    ε0 = 8.85418782 * (10**-12) #m^-3 kg^-1 s^4 A^2
    εr_SiO2 = 3.9
    return εr_SiO2*ε0*gate_voltage/(thickness*abs(fundamental_charge_e)) # C/m^2

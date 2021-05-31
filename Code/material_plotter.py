import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)
from numpy.lib.recfunctions import append_fields
import numpy.polynomial.polynomial as poly
from scipy.stats import linregress
from scipy.optimize import curve_fit
import configparser

#cmap = plt.cm.get_cmap('Set1')
#colors_set1 = cmap(np.linspace(0,1,9))
#colors = colors_set1[[0,3,2,1,4,5]]

fundamental_charge_e = -1.602176634 * (10**-19) # SI: C
ħ = 1.0545718 * 10**-34 #SI m^2 kg / s
kB = 1.38064852 * (10**-23) # m^2 * kg / (s^2 * K)  Boltzmann constant

def_fontsize=10
def_labelsize=10
def_size = 2

cmap = plt.cm.get_cmap('Set1')
colors_set1 = cmap(np.linspace(0,1,9))

def first_occurance_1D(array, val, tol=0.2, starting_index=0):
    itemindex = np.where(abs(array[starting_index:] - val) < abs(tol))
    return itemindex[0][0]

def nth_occurance_1D(array, val, n, tol=0.2, starting_index=0):
    itemindex = np.where(abs(array[starting_index:] - val) < tol)
    return itemindex[0][n-1]

def add_subplot(fig):
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n+1, 1, i+1)
    
    # add the new
    ax = fig.add_subplot(n+1, 1, n+1)
    fig.set_size_inches(8, 6*(n+1))
    return (fig, ax)

def process_file(file):
    print(file)
    if not os.path.isfile(file):
        raise Exception("not a file %s", file)
        return []
    
    """Read in config information from .ini portion"""
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config.read(file)
    except:
        print("failed to read file")
        return
    headerlength = config.getint('Main','Header length',fallback=0)
    
    """get data from file"""
    data = np.genfromtxt(file, skip_header=headerlength-1, names=True, 
                         dtype='<f8', delimiter='\t')
    
    #print(experiment)
    #print(data.dtype.names)
    return data

def save_generic_both(fig, root, filename):
    save_generic_png(fig, root, filename)
    save_generic_svg(fig, root, filename)
    
def save_generic_png(fig, root, filename):
    fig.savefig(os.path.join(root, filename +'.png'), format='png', transparent=True, bbox_inches='tight',pad_inches=.1)
    
def save_generic_svg(fig, root, filename):
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["text.usetex"] = False
    fig.savefig(os.path.join(root, filename +'.svg'), format='svg', transparent=True, bbox_inches='tight',pad_inches=0)

def pretty_plot_single(fig, labels=['',''], color='#000000', yscale='linear', fontsize=10, labelsize=8, labelpad=[0,0]):
    ax = fig.add_subplot(111)
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams["mathtext.default"] = "regular"
    
    ax.set_xlabel(labels[0], fontname="Arial", fontsize=fontsize, labelpad=labelpad[0])
    ax.set_ylabel(labels[1], fontname="Arial", fontsize=fontsize, color=color, labelpad=labelpad[1])
   
    ax.set_yscale(yscale)
    ax.tick_params(bottom=True, top=True, left=True, right=True,
                   which='both', direction='in', labelsize=labelsize, pad=1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    
    ax.tick_params(which='both', width=.5)
    
    ax.minorticks_on()

    if yscale == 'log':
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, numticks=10, subs=subs))

    return ax


def plot_YvsX_generic(Xaxis, Xlabel, Yaxis, Ylabel, XvsYname, \
                      fileroot, files, savename, colors, log=False, size=def_size, majorx=None, \
                      xlim=(None,None), ylim=(None,None), markers=[], \
                      fontsize=def_fontsize, labelsize=def_labelsize, invertaxes=False):    
    for file in files:
        if type(file) is not np.ndarray:
            raise TypeError("Only Numpy arrays allowed. Found: " + str(type(file))) 
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    scale_pow = 1
    scale_label = ''
    
    if not log:
        if isinstance(Yaxis, list):
            Ymax = []
            for yaxis_i in Yaxis:
                Ymax.append([np.nanmax(np.abs(file[yaxis_i])) for file in files])
            Ymax = np.nanmax(Ymax)
        else:
            Ymax = [np.nanmax(np.abs(file[Yaxis])) for file in files]
        (scale_pow, scale_label) = m_order(Ymax)
    try:
        Ylabel = Ylabel % (scale_label) 
    except:
        print('failed to scale y label')
        
    print(scale_label)
    
    ax = pretty_plot_single(fig, labels=[Xlabel, Ylabel],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)
    
    markers += ['.-']*len(colors)

    i = 0
    for file in files:
        if isinstance(Yaxis, list): 
            for yaxis_i in Yaxis:
                ax.plot(file[Xaxis], np.abs(file[yaxis_i]) if log else (file[yaxis_i])*scale_pow,
                    markers[i], ms=3, linewidth=1.5, color=colors[i])
                i = i + 1
        else:
            ax.plot(file[Xaxis], np.abs(file[Yaxis]) if log else (file[Yaxis])*scale_pow,
                    markers[i], ms=3, linewidth=1.5, color=colors[i])
            i = i + 1
    
    if majorx is not None:
        ax.xaxis.set_major_locator(MultipleLocator(majorx))
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if invertaxes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[1], xlim[0])
        if not log:
            ax.set_ylim(ylim[1], ylim[0])
    
    scalename = "_log" if log else "_linear"
    
    print(str(savename)+XvsYname+scalename)   
    
    if savename is None:
        return (fig, ax)
    else:
        save_generic_svg(fig, fileroot, savename+XvsYname+scalename)
        plt.show() 
        plt.clf()
        return None

def m_order(data):
    maxV = np.nanmax(np.abs(data))
    if maxV > 10**12:
        scale = 10**-12
        label = 'T'
    if maxV > 10**9:
        scale = 10**-9
        label = 'G'
    if maxV > 10**6:
        scale = 10**-6
        label = 'M'
    elif maxV > 10**3:
        scale = 10**-3
        label = 'k'
    elif maxV > 1:
        scale = 1
        label = ''
    elif maxV > 10**-3:
        scale = 10**3
        label = 'm'
    elif maxV > 10**-6:
        scale = 10**6
        label = '$\mu$'
    elif maxV > 10**-9:
        scale = 10**9
        label = 'n'
    elif maxV > 10**-12:
        scale = 10**12
        label = 'p'
    else:
        scale = 0
        label = 'SCALE ISSUE'
    return (scale, label)

# cross sections of gating loops
def get_cross_section(fileroot, filenames, increments, V_index):
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    
    indexes = []
    for inc in increments:
        indexes.append([first_occurance_1D(file['Gate_Voltage_V'], inc, starting_index=5)+5 for file in files])
    
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
                             figsize=def_size, xlim=(None, 330), ylim=(None,None), log=True, \
                             fontsize=def_fontsize, labelsize=def_labelsize, colororder=[0,3,2,1,4,5]):
    colors = colors_set1[colororder]
              
    (Currents, Voltages, GateVoltages, Temperatures) = get_cross_section(fileroot, filenames, increments, 1)
    ymax = [10.] if log else np.nanmax([np.nanmax(y) for y in Currents]) 
    (scale_pow, scale_label) = m_order(ymax)
    
    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{I_{D}}$ (%sA)' % scale_label],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)

    for (color, Is) in zip (colors, Currents):
        ax.plot(Temperatures, Is*scale_pow, '.-', ms=3, linewidth=1.5, color=color)
    
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(xlim)
    
    scalename = "_log" if log else "_linear"
    print(savename+  "_loop_I-cross" +scalename)
    save_generic_svg(fig, fileroot, savename + "_loop_I-cross" + scalename)
    plt.show()
    plt.clf()

def plot_loopR_cross_section(fileroot, filenames, savename, increments=[0,25,50,75], \
                             figsize=def_size, xlim=(None, 330), xinc=100, ylim=(None,None), log=True, \
                             fontsize=def_fontsize, labelsize=def_labelsize, colororder=[0,3,2,1,4,5]):
    colors = colors_set1[colororder]
    
    (Currents, Voltages, GateVoltages, Temperatures) = get_cross_section(fileroot, filenames, increments, 1)
    Resistances = []
    for (Vs, Is) in zip(Voltages, Currents):    
        Resistances.append(np.array([v/i for (v,i) in zip(Vs,Is)]))

    ymax = [10.] if log else np.nanmax([np.nanmax(R) for R in Resistances]) 
    (scale_pow, scale_label) = m_order(ymax)

    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{R_{DS}}$ (%sΩ)' % scale_label],
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
    save_generic_svg(fig, fileroot, savename + "_loop_R-cross" + scalename)
    plt.show()
    plt.clf()
    
    
def plot_mobility_μ_cross_section(fileroot, filenames, savename, length, width, increments=[0,25,50,75], \
                             figsize=3, xlim=(None, 330), ylim=(None,None), log=True, fontsize=def_fontsize, labelsize=def_labelsize, colororder=[0,3,2,1,4,5]):
    colors = colors_set1[colororder]
    
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
    (scale_pow, scale_label) = m_order(ymax)
    
    fig = plt.figure(figsize=(figsize, figsize), dpi=300)
    ax = pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{μ}$ (%s $cm^2/V\\cdot s$)' % scale_label],
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
    save_generic_svg(fig, fileroot, savename + "_loop_μ-cross" + scalename)
    plt.show()
    plt.clf()

# generic functions
def plot_IDvsVg_each(fileroot, filenames, savename, log=False, size=def_size, majorx=25,\
                     ylim=(None,None), fontsize=def_fontsize, labelsize=def_labelsize):
   
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    for file in files:
        Iname = '_IDvsVg_' + str(round(file['Temperature_K'][0],1)).zfill(5) + 'K'
        (file0, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
                
        #highpass filter
        ind = np.where(file0['Current_A'] > 2*10**-12)
        file0 = file0[ind]
        
        plot_IDvsVg_generic(fileroot, [file0], savename + Iname, [colors_set1[1]], log=log, \
                              size=size, majorx=majorx, ylim=ylim, fontsize=fontsize, labelsize=labelsize)

      
def plot_IDvsVg_generic(fileroot, files, savename, colors, log=False, size=def_size, majorx=25, 
                        xlim=(None,None), ylim=(None,None), fontsize=def_fontsize, labelsize=def_labelsize):
    return plot_YvsX_generic('Gate_Voltage_V', '$\it{V_{G}}$ (V)', 'Current_A', '$\it{I_{D}}$ (%sA)', '_IDvsVG',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)


def plot_IDvsVg_subleak_generic(fileroot, files, savename, colors, log=False, size=def_size, majorx=25):    
    for file in files:
        file['Current_A'] = file['Current_A']-file['Gate_Leak_Current_A']
        
    return plot_IDvsVg_generic(fileroot, files, savename, colors, log=log, size=size, majorx=majorx)
    
def plot_IleakvsVg_generic(fileroot, files, savename, colors, log=False, adjust=True):    
    fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
    scale_pow = 1
    scale_label = ''
    
    if not log:
        Imax = [np.nanmax(file['Gate_Leak_Current_A']) for file in files]
        (scale_pow, scale_label) = m_order(Imax)
    
    ax = pretty_plot_single(fig, labels=["$\it{V_{G}}$ (V)", '$\it{I_{leak}}$ (%sA)' % (scale_label)],
                             yscale='log' if log else 'linear', fontsize=def_fontsize)
    
    ax.minorticks_on()
    
    occ0 = first_occurance_1D(files[0]['Gate_Voltage_V'], -75.0, tol=0.2, starting_index=0)
    occ1 = first_occurance_1D(files[0]['Gate_Voltage_V'], -75.0, tol=0.2, starting_index=occ0+1)
    files = [files[0][occ0:occ0+occ1+2]]
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Gate_Voltage_V'], np.abs(file['Gate_Leak_Current_A']) if log else (file['Gate_Leak_Current_A'])*scale_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(40))

    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()

def plot_RDSvsVg_generic(fileroot, files, savename, colors=colors_set1, R_ind=1, log=False, size=def_size, majorx=25, ylim=(None,None)):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    R_pow = 1
    R_label = ''
    R_col = 'Resistance_' + str(R_ind) + '_Ohms'
    
    if not log:
        Rmax = [np.nanmax(file[R_col]) for file in files]
        (R_pow, R_label) = m_order(Rmax)
    
    ax = pretty_plot_single(fig, labels=["$\it{V_{G}}$ (V)", '$\it{R_{DS}}$ (%sΩ)' % (R_label)],
                             yscale=('log' if log else 'linear'))
    
    
    for (file, color) in zip(files, colors):
        ax.plot(file['Gate_Voltage_V'], np.abs(file[R_col]) if log else (file[R_col])*R_pow,
                '.-', ms=3, linewidth=1.5, color=color)
    
    ax.xaxis.set_major_locator(MultipleLocator(majorx))
    ax.set_ylim(ylim)
    
    scalename = "_log" if log else "_linear"
    print(savename+scalename)
    save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()
    
# 300K IV plots
def plot_IDvVDS_gating_generic(fileroot, sample, end_name, startindex, incindex, savename, \
                               figsize=def_size, xadj=1, log=False, majorx=None, ylim=None):
    colors = colors_set1[[0,4,3,6,2,8,1]]
    
    start = startindex 
    inc = incindex
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start, start+1*inc)]
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'positive_increasing', colors,\
                         size=figsize, xadj=xadj, ylim=ylim, log=log, majorx=majorx)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+1*inc, start+2*inc)]
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'positive_decreasing', colors[::-1],\
                         size=figsize, xadj=xadj, ylim=ylim, log=log, majorx=majorx)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+2*inc, start+3*inc)]
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'negative_increasing', colors,\
                         size=figsize, invertaxes=True, xadj=xadj, ylim=ylim, log=log, majorx=majorx)
    
    filenames = [(sample + str(i).zfill(3) + end_name) for i in range(start+3*inc, start+4*inc)]
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    plot_IDvsVDS_generic(fileroot, files, savename + 'negative_decreasing', colors[::-1],\
                         size=figsize, invertaxes=True, xadj=xadj, ylim=ylim, log=log, majorx=majorx)

    
def plot_IDvsVDS_generic(fileroot, files, savename, colors, log=False, invertaxes=False, size=def_size, majorx=None, \
                         xadj=0, x_mult=5, fontsize=def_fontsize, labelsize=def_labelsize, xlim=None, ylim=None):    
    return plot_YvsX_generic('Voltage_1_V', '$\it{V_{DS}}$ (V)', 'Current_A', '$\it{I_{D}}$ (%sA)', '_IDvsVDS',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize, invertaxes=invertaxes)
    
def plot_IDvsB_generic(fileroot, files, savename, colors, log=False, symm=False, size=def_size, \
                       xlim=None, ylim=None, majorx=None, fontsize=def_fontsize, labelsize=def_labelsize):
    return plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)', 'Current_A', '$\it{I_{D}}$ (%sA)', '_IDvsB',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)
    
def plot_IDSvsTime_generic(fileroot, files, savename, colors=colors_set1, log=False, size=def_size,\
                           majorx=1800, xlim=(None,None), ylim=(None,None), fontsize=def_fontsize, labelsize=def_labelsize):    
    return plot_YvsX_generic('Time_s', '$\it{t_{s}}$ (V)', 'Current_A', '$\it{I_{D}}$ (%sA)', '_IDvst',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)

def plot_IDvsT_generic(fileroot, files, savename, colors, log=False, size=def_size, majorx=None, 
                        xlim=(None,None), ylim=(None,None), fontsize=def_fontsize, labelsize=def_labelsize):
    return plot_YvsX_generic('Temperature_K', '$\it{T$ (K)', 'Current_A', '$\it{I_{D}}$ (%sA)', '_IDvsT',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)
    
def plot_RSDvsT_generic(fileroot, files, savename, colors, log=False, size=def_size, majorx=None, 
                        xlim=(None,None), ylim=(None,None), fontsize=def_fontsize, labelsize=def_labelsize):
    return plot_YvsX_generic('Temperature_K', '$\it{T}$ (K)', 'Resistance_1_Ohms', '$\it{R_{DS}}$ (%sΩ)', '_RSDvsT',
                      fileroot=fileroot, files=files, savename=savename, colors=colors, log=log, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)
    
def plot_LnRSDvsPowT_generic(fileroot, files, savename, colors, power, power_label, \
                        size=def_size, majorx=None, xlim=(None,None), ylim=(None,None), \
                        fontsize=def_fontsize, labelsize=def_labelsize):

    newfiles = []
    for (file) in (files):
        
        lnR = np.log(file['Resistance_1_Ohms'])
        Tpow = np.power(file['Temperature_K'], power)
        
        file = append_fields(file, 'LnResistance_LnOhms', lnR, np.double, usemask=False)
        file = append_fields(file, 'PowTemperature_K', Tpow, np.double, usemask=False)
        newfiles.append(file)
        
        coeff, stats = poly.polyfit(Tpow, lnR, 1, full = True)
        print("slope: %5f, Intercept: %8f, T0: %8.8f K" % (coeff[1], coeff[0], np.power(coeff[1],-1./power)))
        lnRfit = poly.polyval(Tpow, coeff)
        r2_score, sse, tse = compute_r2_weighted(lnR, lnRfit)
        print("R^2: %5f" % r2_score)
        
        def vrhfit(x, R0, T0):
            return R0 * np.exp((T0 / x)**-power)
        
        popt, pcov = curve_fit(vrhfit, file['Resistance_1_Ohms'], file['Temperature_K'])
        print("exp curve_fit: R0=%5.3f, T0=%5.3f" % tuple(popt))
        
        popt, pcov = curve_fit(vrhfit, file['Resistance_1_Ohms'], file['Temperature_K'], p0=[np.exp(coeff[0]), np.power(coeff[1],-1./power)])
        print("exp-p0 curve_fit: R0=%5.3f, T0=%5.3f" % tuple(popt))

    clean_label = power_label.replace(r"/", "_") 
    return plot_YvsX_generic('PowTemperature_K', '$\it{T^{%s}}$ ($K^{%s}$)' % (power_label, power_label),
                      'LnResistance_LnOhms', '$\it{ln(R)}$ (ln(Ω))', '_logRSDvsT_' + clean_label,
                      fileroot=fileroot, files=newfiles, savename=savename, colors=colors, log=False, size=size, majorx=majorx,
                      xlim=xlim, ylim=ylim, fontsize=fontsize, labelsize=labelsize)

    
def slice_data(file, column, start, end, tol, nth_start=1, nth_finish=1, starting_index=0):
    if start is None:
        occ0 = 0
    else:
        occ0 = nth_occurance_1D(file[column], start, nth_start, tol=tol, starting_index=starting_index)
    if end is None:
        occ1 = np.size(file) - occ0 - 1
    else:
        occ1 = nth_occurance_1D(file[column], end, nth_finish, tol=tol, starting_index=occ0+1)
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


def compute_r2_weighted(y_true, y_pred, weight=None):
    if weight is None:
        weight = np.ones(np.shape(y_true))
    sse = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    tse = (weight * (y_true - np.average(
        y_true, axis=0, weights=weight)) ** 2).sum(axis=0, dtype=np.float64)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse


def process_hall_data(Hall_file, device_width, device_length, device_volt_spacing, device_thickness,
                      T_Rxx_4pt=None,
                      hall_fields=['Voltage_1_V', 'Voltage_2_V'], symmeterize=True, Bfitlimits=(-10,10)):
    
    n2Ds = [] # 2D hall density
    r_squareds = [] # error
    fits = [] 
    μH = [] # mobility
   
    current = Hall_file['Current_A'][0]
    T = Hall_file['Temperature_K'][0]
    
    B_data = Hall_file['Magnetic_Field_T']
    occ0 = first_occurance_1D(B_data, Bfitlimits[0], tol=.05, starting_index=0)
    occ1 = first_occurance_1D(B_data, Bfitlimits[1], tol=.05, starting_index=0)

    # pull R from seperate 4pt data 
    if T_Rxx_4pt is not None:
        #σs = l/(Rxx*w)
        σs = device_volt_spacing / (T_Rxx_4pt * device_width)
    # R from 2pt resistance in actual measurement
    else:
        VDS_data = Hall_file['Voltage_3_V']
        ind = first_occurance_1D(B_data, 0, tol=0.01, starting_index=0)
        R_2pt = VDS_data[ind]/current ##
        σs = device_length / (R_2pt * device_width)
    
    B_data = B_data[occ0:occ1]
    VH_datas = []
    fitdata = []
    
    # hall data for line
    for field in hall_fields:
        V_Hdata = Hall_file[field]
        
        if symmeterize:
            V_Hdata = (V_Hdata - np.flip(V_Hdata))/2
        VH_datas.append(V_Hdata)
        
        V_Hdata = V_Hdata[occ0:occ1]
        
        # fit V_Hall to a line
        (pcoefs, residuals, rank, singular_values, rcond) = \
            np.polyfit(B_data, V_Hdata, 1, full = True)
    
        pfit = np.poly1d(pcoefs)
        fits.append(pfit)
        fitvals = np.empty(np.size(Hall_file['Current_A']))
        fitvals[:] = np.nan
        fitvals[occ0:occ1] = pfit(0) + pfit(1)*B_data
        fitdata.append(fitvals)
    
        # error in fit
        residuals = V_Hdata - pfit(B_data)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((V_Hdata-np.mean(V_Hdata))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squareds.append(r_squared)
    
        # pick a point on the line
        B_point = 1.0 # Tesla
        V_hall0T = pfit.c[1]
        V_hall_T = V_hall0T + B_point*pfit.c[0]
        
        # n2D = B/(Rxy*e) = 1/RH2D*e 
        n2D = B_point*current/((V_hall_T-V_hall0T)*abs(fundamental_charge_e))
        n3D = n2D/device_thickness
        print("%s K: n2D: %s cm^-2, n3D: %s cm^-3" % (round(T,1),
                    np.format_float_scientific(n2D/(100*100), unique=False, precision=5),
                    np.format_float_scientific(n3D/(100*100*100), unique=False, precision=5))
              )
        n2Ds.append(n2D)
        
        # μ = σs/(e*n2D)
        μ = σs/(abs(fundamental_charge_e)*n2D)
        μH.append(μ)
    
        #plot_hall_V_generic(xdata, ydata, '_RvsH_' + str(temp) + 'K_R2_' + str(r_squared), polyfit=pfit)
        print("%s K: μH: %s cm^2/Vs" % (round(T,1), np.multiply(μ,100*100)))
        
        print("Fit R^2: %s" % r_squared)
        
    if len(hall_fields) == 2:
        print("Percent diff: %s%%" % (np.abs((μH[0] - μH[1])/μH[0])*100))
        
        μH.insert(0, (μH[0] + μH[1])/2)
        n2Ds.insert(0, (n2Ds[0] + n2Ds[1])/2)
    
    B_data = Hall_file['Magnetic_Field_T']
    return (B_data, VH_datas, np.array(n2Ds), fits, fitdata, r_squareds, np.array(μH))

def process_MR_data(fileroot, data_file, volt_fields, Bfitlimits=(None,None), plot_data=True, fit_data=True):
    Bfield_data = data_file['Magnetic_Field_T']
    r_squareds = []
    Resistances = []
    ivfitdata = []
    occ0 = 0
    occ1 = np.size(Bfield_data)
    
    if Bfitlimits[0] is not None:
        occ0 = first_occurance_1D(Bfield_data, Bfitlimits[0], tol=Bfitlimits[0]/25, starting_index=0)
    if Bfitlimits[1] is not None:
        occ1 = first_occurance_1D(Bfield_data, Bfitlimits[1], tol=Bfitlimits[1]/25, starting_index=0)
    Bfield_data = Bfield_data[occ0:occ1]
    
    if fit_data:
        for field in volt_fields:
            V_data = data_file[field]
            V_data = V_data[occ0:occ1]
            # fit V_Hall to a line
            (pcoefs, residuals, rank, singular_values, rcond) = \
                np.polyfit(Bfield_data, V_data, 1, full = True)
        
            ivfit = np.poly1d(pcoefs)
            #fits.append(pfit)
            print(ivfit)
            
            V_offset = ivfit.c[1]
            Resistance = ivfit.c[0]
            ivfit = np.poly1d(pcoefs)
            Resistances.append(Resistance)
            
            ivfitvals = np.empty(np.size(data_file['Magnetic_Field_T']))
            ivfitvals[:] = np.nan
            ivfitvals[occ0:occ1] = V_offset + Resistance*Bfield_data
            ivfitdata.append(ivfitvals)
            
        
            # error in fit
            residuals = V_data - (V_offset + Resistance*Bfield_data)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((V_data-np.mean(V_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            r_squareds.append(r_squared)
        
            print("R: %s Ohms" % (np.format_float_scientific(Resistance, unique=False, precision=5)))
            print("Fit R^2: %s" % r_squared)
            
        Ravg = np.average(np.array(Resistance))

    if plot_data:
        numfields = len(volt_fields)
        plot_colors = colors_set1[0:numfields,:]
        markers = ['.-']*numfields + ['-']*numfields
        
        for i, fit in enumerate(ivfitdata): 
            if fit_data:
                data_file = append_fields(data_file, 'MRfit_' + str(i), fit, np.float64, usemask=False)
                plot_colors = np.append(plot_colors, [[0,0,0,1]], axis=0)
                volt_fields.append('MRfit_' + str(i))
            
        return plot_YvsX_generic('Magnetic_Field_T', '$\it{B}$ (T)',
                          volt_fields, '$\it{V}$ (%sV)', '_IvsV-fit_', markers=markers,
                          fileroot=fileroot, files=[data_file], savename="MR", colors=plot_colors, log=False)


def width_Vg(file, current):
    #relevant ranges
    (file1, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', start=-75., end=75., tol=.01, starting_index=0)
    (file2, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', start=75., end=-75., tol=.01, starting_index=0)
    
    Vgs1 = file1['Gate_Voltage_V']
    Is1 = file1['Current_A']
    Vgs2 = file2['Gate_Voltage_V']
    Is2 = file2['Current_A']
    
    if (abs(file1['Gate_Voltage_V'][0] - -75) > .2 or 
        abs(file1['Gate_Voltage_V'][-1] - 75) > .2 or 
        abs(file2['Gate_Voltage_V'][0] - 75) > .2 or 
        abs(file2['Gate_Voltage_V'][-1] - -75) > .2):
        print("Something went wrong with bounds")
        return np.nan
    
    if len(Vgs1) == 0 or len(Vgs2) == 0:
        return np.nan
    
    Vgs1_index1 = 0
    for i in range(len(file1['Current_A'])-3):
        if (file1['Current_A'][i] > current and 
            file1['Current_A'][i+1] > current and
            file1['Current_A'][i+2] > current and
            file1['Current_A'][i+3] > current):
            break
        Vgs1_index1 = i
        
    Vgs2_index1 = 0
    for i in range(len(file2['Current_A'])-3):
        if (file2['Current_A'][i] < current and 
            file2['Current_A'][i+1] < current and
            file2['Current_A'][i+2] < current and
            file2['Current_A'][i+3] < current):
            break
        Vgs2_index1 = i
    
    if Vgs1_index1 == 0 or Vgs1_index1+1 >= len(Vgs1) or Vgs2_index1 == 0 or Vgs1_index1+1 >= len(Vgs2):
        return np.nan
    
    m1 = (Is1[Vgs1_index1+1] - Is1[Vgs1_index1]) / (Vgs1[Vgs1_index1+1] - Vgs1[Vgs1_index1])
    b1 = Is1[Vgs1_index1] - m1*Vgs1[Vgs1_index1]
    
    m2 = (Is2[Vgs2_index1+1] - Is2[Vgs2_index1]) / (Vgs2[Vgs2_index1+1] - Vgs2[Vgs2_index1])
    b2 = Is2[Vgs2_index1] - m2*Vgs2[Vgs2_index1]
    
    
    # find the points just above and below the given current
    #Vgs1_index1 = np.argmin(np.abs(Is1 - current))
    #if Vgs1_index1 == 0 or Vgs1_index1+1 >= len(Vgs1):
    #    return np.nan
    
    #if (Is1[Vgs1_index1] - current < 0 and Is1[Vgs1_index1+1] - current > 0) or \
    #    (Is1[Vgs1_index1] - current > 0 and Is1[Vgs1_index1+1] - current < 0):
    #    Vgs1_index2 = Vgs1_index1+1
    #else:
    #    Vgs1_index2 = Vgs1_index1-1
    
    #Vgs2_index1 = np.argmin(np.abs(Is2 - current))
    #if Vgs2_index1 == 0 or Vgs2_index1+1 >= len(Vgs2):
    #    return np.nan
    
    #if (Is2[Vgs2_index1] - current < 0 and Is2[Vgs2_index1+1] - current > 0) or \
    #    (Is2[Vgs2_index1] - current > 0 and Is2[Vgs2_index1+1] - current < 0):
    #    Vgs2_index2 = Vgs2_index1+1
    #else:
    #    Vgs2_index2 = Vgs2_index1-1

    # fit the line formula, y = mx + b 
    #m1 = (Is1[Vgs1_index2] - Is1[Vgs1_index1]) / (Vgs1[Vgs1_index2] - Vgs1[Vgs1_index1])
    #b1 = Is1[Vgs1_index1] - m1*Vgs1[Vgs1_index1]
    
    #m2 = (Is2[Vgs2_index2] - Is2[Vgs2_index1]) / (Vgs2[Vgs2_index2] - Vgs2[Vgs2_index1])
    #b2 = Is2[Vgs2_index1] - m2*Vgs2[Vgs2_index1]
    
    # calc VG at current value
    VG1 = (current - b1)/m1
    VG2 = (current - b2)/m2
    
    #if np.abs(VG2 - VG1) > 150:
    #    print("VG1 %s %s" % (Vgs1[Vgs2_index1], Vgs1[Vgs1_index2]))
    #    print("VG2 %s %s" % (Vgs2[Vgs2_index1], Vgs2[Vgs1_index2]))
    #    print("Current1 %s %s" % (Is1[Vgs2_index1], Is1[Vgs1_index2]))
    #    print("Current2 %s %s" % (Is2[Vgs2_index1], Is2[Vgs1_index2]))
        
    
    return np.abs(VG2 - VG1)

def calc_minSS(fileroot, file, Npoints=4, subplot=True, startend=-75, switch=75, Icutoff=5*10**-11):    
    # split relevant ranges
    (file1, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', start=startend, end=switch, tol=.01, starting_index=0)
    (file2, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', start=switch, end=startend, tol=.01, starting_index=0)
    (fileall, start_ind, end_ind) = slice_data(file, 'Gate_Voltage_V', start=startend, end=startend, tol=.01, starting_index=0)
    
    
    Vgs1 = file1['Gate_Voltage_V']
    Is1 = file1['Current_A']
    Vgs2 = file2['Gate_Voltage_V']
    Is2 = file2['Current_A']
    
    # make sure it just didn't go to the ends
    if (abs(file1['Gate_Voltage_V'][0] - -75) > .2 or 
        abs(file1['Gate_Voltage_V'][-1] - 75) > .2 or 
        abs(file2['Gate_Voltage_V'][0] - 75) > .2 or 
        abs(file2['Gate_Voltage_V'][-1] - -75) > .2):
        print("Something went wrong with bounds")
        return np.nan
    
    # min size of split
    if len(Vgs1) <= Npoints or len(Vgs2) <= Npoints:
        return np.nan
    
    # log10(ID)
    logI1 = np.log10(np.abs(file1['Current_A']))
    logI2 = np.log10(np.abs(file2['Current_A']))
    Icutoff = np.log10(Icutoff)
    
    SS1 = []
    SS2 = []
    SS1fitb = []
    SS2fitb = []
    
    for i in range(len(logI1)-Npoints):
        # to be fit
        subVG = Vgs1[i:i+Npoints]
        subI = logI1[i:i+Npoints]
        
        # fit
        (pcoefs, residuals, rank, singular_values, rcond) = \
                np.polyfit(subVG, subI, 1, full = True)
        
        ivfit = np.poly1d(pcoefs)
        slope = ivfit.c[0]
        if np.any(subI < Icutoff):
            slope = np.NINF
        #SS = 1/slope # http://wla.berkeley.edu/~ee40/fa03/lecture/lecture23.pdf
        SS1.append(slope)
        SS1fitb.append(ivfit.c[1])
        
    for i in range(len(logI2)-Npoints):
        # to be fit
        subVG = Vgs2[i:i+Npoints]
        subI = logI2[i:i+Npoints]
        
        # fit
        (pcoefs, residuals, rank, singular_values, rcond) = \
                np.polyfit(subVG, subI, 1, full = True)
        
        ivfit = np.poly1d(pcoefs)
        slope = ivfit.c[0]
        if np.any(subI < Icutoff):
            slope = np.NINF
        SS2.append(slope)
        SS2fitb.append(ivfit.c[1])
        
    SS1min = np.argmax(SS1)
    SS2min = np.argmax(SS2)
    SS1 = 1/np.array(SS1)
    SS2 = 1/np.array(SS2)
    
    if subplot:
        fig, ax = plot_IDvsVg_generic(fileroot, [fileall], None, [colors_set1[1]], log=True, size=2, majorx=25,
                              ylim=(None,None), fontsize=10, labelsize=10)
        # add fit 1
        VGs = np.array([Vgs1[SS1min], Vgs1[SS1min+Npoints-1]])
        b = SS1fitb[SS1min]
        m = 1/SS1[SS1min]
        Is = 10**(VGs*m+b)
        ax.plot(VGs, Is, '-', ms=1, linewidth=1.5, color=colors_set1[0])
    
        #add fit 2
        VGs = np.array([Vgs2[SS2min], Vgs2[SS2min+Npoints-1]])
        b = SS2fitb[SS2min]
        m = 1/SS2[SS2min]
        Is = 10**(VGs*m+b)
        ax.plot(VGs, Is, '-', ms=1, linewidth=1.5, color=colors_set1[2])
        plt.show() 
        plt.clf()
        
    return (SS1[SS1min], Vgs1[SS1min], Vgs1[SS1min+Npoints], SS2[SS2min], Vgs2[SS2min], Vgs2[SS2min+Npoints])


##### UNUSED

def plot_ΔVGvT(fileroot, filenames, current, size=def_size, log=False):
    savename = '_DVGvT'
    size = 2
    colors = colors_set1

    files = [process_file(fileroot + x) for x in filenames]
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{ΔV_{G}}$ (V)'],
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
    save_generic_svg(fig, fileroot, savename+scalename)
    plt.show()
    plt.clf()

def plot_maxSS_vs_T(fileroot, filenames, savename, size=2, showthreshold=False, subplot=True, Npoints=4, Icutoff=5*10**-11):
    files = [process_file(os.path.join(fileroot, x)) for x in filenames]
    
    temperatures = []
    SSinc = []
    SSdec = []
    
    for file in files:
        temperature = file['Temperature_K'][0]
        print("Temperature %s K" % str(temperature))
        temperatures.append(temperature)
        SSi, Vgsi1, Vgsi2, SSd, Vgsd1, Vgsd2 = calc_minSS(fileroot, file, Npoints=Npoints,
                                                          subplot=subplot, Icutoff=Icutoff)
        SSinc.append(SSi)
        SSdec.append(SSd)
    
    fig = plt.figure(figsize=(size, size), dpi=300)
    ax = pretty_plot_single(fig, labels=["$\it{T}$ (K)", '$\it{SS_{min}}$ (V/dec)'],
                             yscale='linear', fontsize=10, labelsize=10)
    
    ax.plot(temperatures, SSinc, '.-', ms=3, linewidth=1.5, color=colors_set1[0])
    ax.plot(temperatures, SSdec, '.-', ms=3, linewidth=1.5, color=colors_set1[2])
    
    print(SSinc)
    print(SSdec)
    
    # add threshold
    if showthreshold:
        threshold = np.log(10)*kB*np.array(temperatures)/np.abs(fundamental_charge_e)
        ax.plot(temperatures, threshold, '.-', ms=3, linewidth=1.5, color=colors_set1[1])
    
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(0, 322)
    
    if savename is None:
        return (fig, ax)
    else:
        save_generic_svg(fig, fileroot, savename)
        plt.show() 
        plt.clf()
        return None

def process_R_4pt(RTloop_2_4pt_filenames):
    #seperate files for R_4pt
    (cs_Currents_left, cs_Voltages_left, _, cs_Temperatures) = \
        get_cross_section(fileroot, RTloop_2_4pt_filenames, [75.], 1)
    (cs_Currents_right, cs_Voltages_right, _, cs_Temperatures) = \
        get_cross_section(fileroot, RTloop_2_4pt_filenames, [75.], 2)
        
    occ0 = first_occurance_1D(cs_Temperatures, T, tol=.2, starting_index=0)
    R_left = cs_Voltages_left[0][occ0]/cs_Currents_left[0][occ0]
    R_right = cs_Voltages_right[0][occ0]/cs_Currents_right[0][occ0]
    #print("R_left: %s Ω and R_right %s Ω" % (round(R_left), round(R_right)))
    Rxx_4pt = (R_left+R_right)/2


def process_IV_data(fileroot, data_file, volt_fields, Ilimits=(None,None), plot_data=True):
    current_data = data_file['Current_A']
    r_squareds = []
    Resistances = []
    ivfitdata = []
    occ0 = 0
    occ1 = np.size(current_data)
    
    if Ilimits[0] is not None:
        occ0 = first_occurance_1D(current_data, Ilimits[0], tol=max(Ilimits[0]/25, 1e-10), starting_index=0)
    if Ilimits[1] is not None:
        occ1 = first_occurance_1D(current_data, Ilimits[1], tol=max(Ilimits[1]/25, 1e-10), starting_index=0)
    current_data = current_data[occ0:occ1]
    
    for field in volt_fields:
        V_data = data_file[field]
        V_data = V_data[occ0:occ1]
        # fit V_Hall to a line
        (pcoefs, residuals, rank, singular_values, rcond) = \
            np.polyfit(current_data, V_data, 1, full = True)
    
        ivfit = np.poly1d(pcoefs)
        #fits.append(pfit)
        print(ivfit)
        
        V_offset = ivfit.c[1]
        Resistance = ivfit.c[0]
        ivfit = np.poly1d(pcoefs)
        Resistances.append(Resistance)
        
        ivfitvals = np.empty(np.size(data_file['Current_A']))
        ivfitvals[:] = np.nan
        ivfitvals[occ0:occ1] = V_offset + Resistance*current_data
        ivfitdata.append(ivfitvals)
        
    
        # error in fit
        residuals = V_data - (V_offset + Resistance*current_data)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((V_data-np.mean(V_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squareds.append(r_squared)
    
        print("R: %s Ohms" % (np.format_float_scientific(Resistance, unique=False, precision=5)))
        print("Fit R^2: %s" % r_squared)
        
    Ravg = np.average(np.array(Resistance))

    if plot_data:
        numfields = len(volt_fields)
        plot_colors = colors_set1[0:numfields,:]
        markers = ['.-']*numfields + ['-']*numfields
        
        for i, fit in enumerate(ivfitdata): 
            data_file = append_fields(data_file, 'IVfit_' + str(i), fit, np.float64, usemask=False)
            plot_colors = np.append(plot_colors, [[0,0,0,1]], axis=0)
            volt_fields.append('IVfit_' + str(i))
            
        return plot_YvsX_generic('Current_A', '$\it{I}$ (A)',
                          volt_fields, '$\it{V}$ (%sV)', '_IvsV-fit_', markers=markers,
                          fileroot=fileroot, files=[data_file], savename=None, colors=plot_colors, log=False)
    
    return (Ravg, Resistances, r_squareds)


def conducting_range(file):
    return [(gv, i) for (gv, i) in zip(file['Gate_Voltage_V'], file['Current_A']) if i > 10**-10]

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
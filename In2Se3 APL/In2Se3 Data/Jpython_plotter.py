# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:08:17 2019

@author: Justin
"""

import numpy as np
import os
import configparser
import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, AutoMinorLocator)
import matplotlib.ticker


colors_hot_cold = ['#0000FF', '#2A00D4', '#5500AA', '#7F00AA', '#AA0055', '#D4002A', '#FF0000', '#000000']
colors_mathem = ['#5E81B5', '#e19c24', '#8fb032', '#eb6235', '#8778b3', 
                 '#5d9ec7', '#ffbf00', '#a5609d', '#929600', '#e95536', '#6685d9', '#c56e1a', 
                 '#f89f13', '#bc5b80', '#47b66d']
cmap = plt.cm.get_cmap('Set1')
colors_set1 = cmap(np.linspace(0,1,9))

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
    experiment = config.get('Main','Experiment',fallback='')
    
    """plot data from file"""
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
    fig.savefig(os.path.join(root, filename +'.svg'), format='svg', transparent=True, bbox_inches='tight',pad_inches=0)
    
def pretty_plot(fig, labels=['',''], colors=['#000000','#000000'], yscale=['linear','linear'], fontsize=12):
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(labels[0], fontname="Arial", fontsize=fontsize)
    ax.set_ylabel(labels[1], fontname="Arial", fontsize=fontsize, color=colors[0])
    
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax2 = ax.twinx()
    ax3 = ax2.twiny()
    if len(labels) < 3:
        ax2.set_yscale(ax.get_yscale())
        ax2.set_ylim(ax.get_ylim())
        ax2.tick_params(which='both', direction='in', labelright=False)
    else:
        ax2.set_ylabel(labels[2], fontname="Arial", fontsize=fontsize, color=colors[1])
        ax2.tick_params(which='both', direction='in', labelright=True)
         
    
    ax2.minorticks_on()
    if len(labels) < 4:
        ax3.set_xscale(ax.get_xscale())
        ax3.set_xlim(ax.get_xlim())
        ax3.tick_params(which='both', direction='in', labeltop=False)
    else:
        ax2.tick_params(which='both', direction='in', labeltop=True)
    
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax.set_yscale(yscale[0])
    ax2.set_yscale(yscale[1])
    ax3.set_yscale(yscale[1])

    return (ax, ax3, ax2)

def pretty_plot_single(fig, labels=['',''], color='#000000', yscale='linear', fontsize=10, labelsize=8, labelpad=[0,0]):
    ax = fig.add_subplot(111)
    
    plt.rcParams["font.family"] = "Arial"
    
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
        nticks = 10 # up to fix skipped minor ticks
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=nticks))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, numticks=nticks, subs=subs))
    
    new_rc_params = {'text.usetex': False,
                     "svg.fonttype": 'none'
                     }
    plt.rcParams.update(new_rc_params)
        
    return ax

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

def m_order(data):
    maxV = np.nanmax(np.abs(data))
    if maxV > 10**12:
        scale = 10**-12
        label = 'T'
    elif maxV > 10**9:
        scale = 10**-9
        label = 'G'
    elif maxV > 10**6:
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
        label = 'μ'
    elif maxV > 10**-9:
        scale = 10**9
        label = 'n'
    else:
        scale = 10**12
        label = 'p'
    return (scale, label)

def first_occurance_1D(array, val, tol=0.2, starting_index=0):
    itemindex = np.where(abs(array[starting_index:] - val) < tol)
    return itemindex[0][0]

def nth_occurance_1D(array, val, n, tol=0.2, starting_index=0):
    itemindex = np.where(abs(array[starting_index:] - val) < tol)
    return itemindex[0][n-1]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 23:15:45 2016

@author: Justin_Rodriguez
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import configparser
import os
import argparse
import glob
import re

def dict_key_match(d, filter_string):
    keys = []
    for key in d.dtype.names:
        if key.startswith(filter_string):
            keys.append(key)
    return keys

def add_subplot(fig):
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n+1, 1, i+1)
    
    # add the new
    ax = fig.add_subplot(n+1, 1, n+1)
    fig.set_size_inches(8, 6*(n+1))
    return (fig, ax)

def remove_outliers(column1, column2):
    """Removes any outlier data from both columns of data. A point is 
    considered an outlier if it is outside 3 STD from the average of the second
    column of data"""
    avg1 = np.nanmean(column1)
    std1 = np.nanstd(column1)
    avg2 = np.nanmean(column2)
    std2 = np.nanstd(column2)
    ind = np.where(np.logical_and(np.logical_and(np.abs(column1-avg1) < 3*std1,
                                                  np.abs(column2-avg2) < 3*std2),
                                  np.isfinite(column1+column2)))
    
    datalen = len(column1)
    indlen = len(ind[0])
    if datalen != indlen:
        print(str(datalen-indlen) + " point(s) were ignored as outliers")
    return (column1[ind], column2[ind], datalen-indlen)

def plot_generic_IV(data, filename, is_V_biased, X_header, X_label, X_unit):
 
    r = re.compile("Voltage_[1-9]_V")
    count = len(list(filter(r.match, data.dtype.names)))
    
    Y_header = 'Current_A' if is_V_biased else 'Resistance_' + str(count) + '_Ohms'
    Y_label = 'Current' if is_V_biased else '2pt Resistance'
    Y_unit = 'A' if is_V_biased else "\N{OHM SIGN}"
    
        
    """Plots X vs Resistance|Current data"""
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    (xdata, ydata, outliers) = remove_outliers(data[X_header], data[Y_header])

    ax.plot(xdata, ydata,'bo')    
    ax.set_title(os.path.basename(filename) +
                       ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.xaxis.set_major_formatter(EngFormatter(unit=X_unit))
    ax.yaxis.set_major_formatter(EngFormatter(unit=Y_unit))
    
    return fig
    
def plot_generic_additionalR(data, is_V_biased, fig, X_header, X_label, X_unit):
    r = re.compile("Voltage_[1-9]_V")
    count = len(list(filter(r.match, data.dtype.names)))
    
    if is_V_biased:
        additional = range(2, count+1)
    else:
        additional = range(1, count)
    
    for i in additional:
        (fig, ax) = add_subplot(fig)
    
        (xdata, ydata, outliers) = remove_outliers(data[X_header], 
                                        data["Resistance_" + str(i) +"_Ohms"])
        ax.plot(xdata, ydata,'bo')    
        ax.set_xlabel(X_label)
        ax.set_ylabel("Additional Resistance #" + str(i))
        ax.xaxis.set_major_formatter(EngFormatter(unit=X_unit))
        ax.yaxis.set_major_formatter(EngFormatter(unit="\N{OHM SIGN}"))
    
    return fig

def save_generic(fig, filename):
    fig.savefig(os.path.join(os.path.dirname(filename),
                       os.path.splitext(os.path.basename(filename))[0]
                        +'.png'), format='png')
    
def plot_VvI(data, filename):
    """Plots Current vs Voltage data"""
    volts = dict_key_match(data, "Voltage")
    vlength = len(volts)
    
    fig = plt.figure(figsize=(8, 6*vlength), dpi=300)
    
    for idx, volt in enumerate(volts):
        ax = fig.add_subplot(100*vlength + 10 + (idx+1))
        
        (xdata, ydata, outliers) = remove_outliers(data['Current_A'], data[volt])
        ax.plot(xdata, ydata,'bo')    
        ax.set_title(os.path.basename(filename) +
                           ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
        ax.set_xlabel('Current')
        ax.set_ylabel(volt)
        ax.xaxis.set_major_formatter(EngFormatter(unit='A'))
        ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
    
    fig = plot_generic_additionalR(data, False, fig, 'Current_A', 'Current', 'A')
    
    save_generic(fig, filename)
    
def plot_IvV(data, filename):
    """Plots Current vs Voltage data"""
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    (xdata, ydata, outliers) = remove_outliers(data['Voltage_1_V'], data['Current_A'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_title(os.path.basename(filename) +
                       ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    ax.xaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='A'))
    
    fig = plot_generic_additionalR(data, True, fig, 'Voltage_1_V', 'Voltage', 'V')
    save_generic(fig, filename)

def plot_RvVg(data, filename):
    """Plots Gate Voltage vs Current data"""
    is_V_biased = data.dtype.names[4] == "Current_A"
    fig = plot_generic_IV(data, filename, is_V_biased, 'Gate_Voltage_V', 'Gate Voltage', 'V')
    fig = plot_generic_additionalR(data, is_V_biased, fig, 'Gate_Voltage_V', 'Gate Voltage', 'V')
    
    (fig, ax) = add_subplot(fig)
    (xdata, ydata, outliers) = remove_outliers(data['Gate_Voltage_V'], data['Gate_Leak_Current_A'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_xlabel('Gate Voltage')
    ax.set_ylabel('Leak Current')
    ax.xaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.yaxis.set_major_formatter(EngFormatter(unit="A"))
    save_generic(fig, filename)

def plot_RvT(data, filename):
    """Plots Temperature vs Resistance data"""
    is_V_biased = data.dtype.names[3] == "Current_A"
    fig = plot_generic_IV(data, filename, is_V_biased, 'Temperature_K', 'Temperature', 'K')
    fig = plot_generic_additionalR(data, is_V_biased, fig, 'Temperature_K', 'Temperature', 'K')
    save_generic(fig, filename)

def plot_RvH(data, filename):
    """Plots Gate Voltage vs Resistance data"""
    is_V_biased = data.dtype.names[3] == "Current_A"
    fig = plot_generic_IV(data, filename, is_V_biased, 'Magnetic_Field_G', 'Magnetic Field', 'G')
    fig = plot_generic_additionalR(data, is_V_biased, fig, 'Magnetic_Field_G', 'Magnetic Field', 'G')
    save_generic(fig, filename)
    
def plot_Rvtime(data, filename):
    """Plots Temperature vs Resistance data"""
    is_V_biased = data.dtype.names[2] == "Current_A"
    fig = plot_generic_IV(data, filename, is_V_biased, 'Time_s', 'Time', 's')
    fig = plot_generic_additionalR(data, is_V_biased, fig, 'Time_s', 'Temperature', 's')
    save_generic(fig, filename)

def process_file(file):
    if not os.path.isfile(file):
        print("not a file")
        return
    
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
    
    print(data.dtype.names)
    
    if re.match('RvsT', experiment.strip('"')):
        plot_RvT(data, file)
    elif re.match('VvsI', experiment.strip('"')):
        plot_VvI(data, file)
    elif re.match('IvsV', experiment.strip('"')):
        plot_IvV(data, file)
    elif re.match('RvsH', experiment.strip('"')):
        plot_RvH(data, file)
    elif re.match('RvsVg', experiment.strip('"')):
        plot_RvVg(data, file)
    elif re.match('Rvst', experiment.strip('"')):
        plot_Rvtime(data, file)
        
def process_directory(dirpath):
    if not os.path.isdir(dirpath):
        print("not a directory")
        return
    
    for root, dirs, files in os.walk(dirpath, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".txt" \
               and "detailed" not in name:
                   print(os.path.join(root, name))
                   try:
                       process_file(os.path.join(root, name))
                   except Exception as e:
                       print(e)

def main():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Process image file for data')
    parser.add_argument('filename', metavar='filename', type=str, nargs=1,
                       help='The file to plot.')
    args = parser.parse_args()

    """Process file"""
    filename = args.filename[0]
    files = glob.glob(filename)
    for file in files:
        print(file)
        if os.path.isdir(file):
            process_directory(file)
        else:
            process_file(file)

if __name__== "__main__":
    main()        


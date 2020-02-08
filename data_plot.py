# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:58:51 2018

@author: Justin
"""
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
from mpl_toolkits.mplot3d import Axes3D

def dict_key_match(d, filter_string):
    keys = []
    for key in d.dtype.names:
        if key.startswith(filter_string):
            keys.append(key)
    return keys

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
    
def plot_RvT(data, filename):
    """Plots Temperature vs Resistance data"""
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    (xdata, ydata, outliers) = remove_outliers(data['Temperature_K'], data['Resistance_1_Ohms'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_title(os.path.basename(filename) +
                       ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Resistance')
    ax.xaxis.set_major_formatter(EngFormatter(unit='K'))
    ax.yaxis.set_major_formatter(EngFormatter(unit="\N{OHM SIGN}"))
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
        
    fig.savefig(os.path.join(os.path.dirname(filename),
                       os.path.splitext(os.path.basename(filename))[0]
                        +'.png'), format='png')
    
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
    fig.savefig(os.path.join(os.path.dirname(filename),
                       os.path.splitext(os.path.basename(filename))[0]
                        +'.png'), format='png')


def plot_RvVg(data, filename):
    """Plots Gate Voltage vs Current data"""
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(211)
    
    (xdata, ydata, outliers) = remove_outliers(data['Gate_Voltage_V'], data['Current_A'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_title(os.path.basename(filename) +
                       ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
    ax.set_xlabel('Gate Voltage')
    ax.set_ylabel('Current')
    ax.xaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.yaxis.set_major_formatter(EngFormatter(unit="A"))
    
    ax = fig.add_subplot(212)
    (xdata, ydata, outliers) = remove_outliers(data['Gate_Voltage_V'], data['Gate_Leak_Current_A'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_xlabel('Gate Voltage')
    ax.set_ylabel('Leak Current')
    ax.xaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.yaxis.set_major_formatter(EngFormatter(unit="A"))
    fig.savefig(os.path.join(os.path.dirname(filename),
                       os.path.splitext(os.path.basename(filename))[0]
                        +'.png'), format='png')


def plot_RvH(data, filename):
    """Plots Gate Voltage vs Resistance data"""
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    (xdata, ydata, outliers) = remove_outliers(data['Magnetic_Field_G'], data['Resistance_Ohms'])
    ax.plot(xdata, ydata,'bo')    
    ax.set_title(os.path.basename(filename) +
                       ('' if outliers == 0 else ' (' + str(outliers) + ' outliers)'))
    ax.set_xlabel('Magnetic Field')
    ax.set_ylabel('Resistance')
    ax.xaxis.set_major_formatter(EngFormatter(unit='G'))
    ax.yaxis.set_major_formatter(EngFormatter(unit="\N{OHM SIGN}"))
    fig.savefig(os.path.join(os.path.dirname(filename),
                       os.path.splitext(os.path.basename(filename))[0]
                        +'.png'), format='png')

def process_file(file):
    if os.path.isfile(file):
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
        
    return (data.dtype.names, data)

def main():
    #loops
    file300 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_026_RvsVg_loops.txt'
    file250 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_027_RvsVg_loops.txt'
    file200 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_028_RvsVg_loops.txt'
    file150 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_029_RvsVg_loops.txt'
    file100 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_030_RvsVg_loops.txt'
    file050 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_031_RvsVg_loops.txt'
    file004 = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_032_RvsVg_loops.txt'
    
    (names300, data300) = process_file(file300)
    (names250, data250) = process_file(file250)
    (names200, data200) = process_file(file200)
    (names150, data150) = process_file(file150)
    (names100, data100) = process_file(file100)
    (names050, data050) = process_file(file050)
    (names004, data004) = process_file(file004)
    
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.log10(data300['Resistance_2_Ohms'])
    ax.scatter(data300['Temperature_K'], data300['Gate_Voltage_V'], np.where(True, x, np.NaN), c='#ff0000', marker='o', label='300K')
    x = np.log10(data250['Resistance_2_Ohms'])
    ax.scatter(data250['Temperature_K'], data250['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#d4002a', marker='o', label='250K')
    x = np.log10(data200['Resistance_2_Ohms'])
    ax.scatter(data200['Temperature_K'], data200['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#aa0055', marker='o', label='200K')
    x = np.log10(data150['Resistance_2_Ohms'])
    ax.scatter(data150['Temperature_K'], data150['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#7f00aa', marker='o', label='150K')
    x = np.log10(data100['Resistance_2_Ohms'])
    ax.scatter(data100['Temperature_K'], data100['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#5500aa', marker='o', label='100K')
    x = np.log10(data050['Resistance_2_Ohms'])
    ax.scatter(data050['Temperature_K'], data050['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#2a00d4', marker='o', label='050K')
    x = np.log10(data004['Resistance_2_Ohms'])
    ax.scatter(data004['Temperature_K'], data004['Gate_Voltage_V'], np.where(x<10, x, np.NaN), c='#0000ff', marker='o', label='004K')
               
    # cool           
    file75V = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_018_RvsT_75V.txt'
    file0Vup = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_020_RvsT_0V-up.txt'
    file0Vdown = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_025_RvsT.txt'
    file_75V = r'C:\Users\jrr330_admin\Box\Liu Lab\Data\Dilution-PPMS-Dip-He3\Justin\In2Se3\JR190919\JR190919_04\JR190919_04_022_RvsT_-75V.txt'
               
    (names75V, data75V) = process_file(file75V)
    (names0Vup, data0Vup) = process_file(file0Vup)
    (names0Vdown, data0Vdown) = process_file(file0Vdown)
    (names_75V, data_75V) = process_file(file_75V)
    
    x = np.log10(data75V['Resistance_2_Ohms'])
    ax.scatter(data75V['Temperature_K'], data75V['Gate_Voltage_V'], np.where(True, x, np.NaN), c='#393e41', marker='o', label='75V')
    x = np.log10(data0Vup['Resistance_2_Ohms'])
    ax.scatter(data0Vup['Temperature_K'], data0Vup['Gate_Voltage_V'], np.where(True, x, np.NaN), c='#16db93', marker='o', label='0V+')
    x = np.log10(data0Vdown['Resistance_2_Ohms'])
    ax.scatter(data0Vdown['Temperature_K'], data0Vdown['Gate_Voltage_V'], np.where(True, x, np.NaN), c='#efea5a', marker='o', label='0V+')
    #x = np.log10(data_75V['Resistance_2_Ohms'])
    #ax.scatter(data_75V['Temperature_K'], data_75V['Gate_Voltage_V'], np.where(True, x, np.NaN), c='#f29e4c', marker='o', label='-75V')
        
    ax.set_xlabel('Temperature (K)', fontname="Arial", fontsize=12)
    ax.set_ylabel('Gate Voltage (V)', fontname="Arial", fontsize=12)
    ax.set_zlabel('Log(R) (log(Ohm))', fontname="Arial", fontsize=12)
    
    ax.view_init(5, 45)
    ax.legend()

    plt.show()

if __name__== "__main__":
  main()

            


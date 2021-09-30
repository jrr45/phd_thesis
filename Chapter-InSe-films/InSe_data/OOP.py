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


OOP_1 = mp.capacitor_device()
OOP_1.name = 'OOP_1'
OOP_1.fileroot = os.path.join('Out-of-plane', '2021-09-06 InSe')
OOP_1.thickness = 30.e-9
OOP_1.width = 50.e-6
OOP_1.length = 50.e-6

OOP_2 = mp.capacitor_device()
OOP_2.name = 'OOP_2'
OOP_2.fileroot = os.path.join('Out-of-plane', '2021-09-05 InSe')

OOP_3 = mp.capacitor_device()
OOP_3.name = 'OOP_3'
OOP_3.fileroot = os.path.join('Out-of-plane', '2021-08-28 InSe')

OOP_4 = mp.capacitor_device()
OOP_4.name = 'OOP_4'
OOP_4.fileroot = os.path.join('Out-of-plane', '2021-09-26 InSe')

def plot_rho_vs_T_generic(device, filename, data_column_names, plot_column_names, 
                          size=2, savename=''):
    dataset = mp.process_capacitor_file(device, filename, data_column_names)

    for (RoI, Measurement, data) in dataset:
        Temperature = data['Temperature_C']

        if RoI == 'Real' and Measurement == 'ZTD0':
            ylabel = 'Z (%sΩ)'
            XvsYname = '_ZvsT_'
        elif RoI == 'Imag' and Measurement == 'ZTD0':
            ylabel = 'φ (%sdeg)'
            XvsYname = '_PhasevsT_'
        elif RoI == 'Real' and Measurement == 'CD0':
            ylabel = 'C (%sF)'
            XvsYname = '_CvsT_'
        elif RoI == 'Imag' and Measurement == 'CD0':
            ylabel = 'tan(δ)'
            XvsYname = '_LTvsT_'
        else:
            print('Could not match one of the data patterns')
            continue
        
        data = append_fields(data, 'Temperature_K', Temperature+273, np.double, usemask=False)
        
        mp.plot_YvsX_generic('Temperature_K', 'T (K)', plot_column_names, 
                             ylabel, XvsYname, device=device, files=[data], 
                             savename=(device.name+savename), colors=mp.colors_set1, 
                             log=False, size=size)


def plot_rho_vs_T_OOP_1(size=2):
    device = OOP_1
    filename = '2021-09-06 C-Z_1.txt'
    data_column_names = ['Temperature_C', '1kHz', '10kHz', '100kHz', '1MHz']
    plot_column_names = ['10kHz', '100kHz', '1MHz']
    
    plot_rho_vs_T_generic(device, filename, data_column_names, 
                          plot_column_names, size=size)
    
def plot_rho_vs_T_OOP_2(size=2):
    device = OOP_2
    filename = '2021-09-05 C-Z_1.txt'
    data_column_names = ['Temperature_C', '1kHz', '10kHz', '100kHz', '1MHz']
    plot_column_names = ['10kHz', '100kHz', '1MHz']
    
    plot_rho_vs_T_generic(device, filename, data_column_names, 
                          plot_column_names, size=size)
    
def plot_rho_vs_T_OOP_3(size=2):
    device = OOP_3
    filename = '2021-08-28 C-Z_1.txt'
    data_column_names = ['Temperature_C', '1kHz', '10kHz', '100kHz', '1MHz']
    plot_column_names = ['10kHz', '100kHz', '1MHz']
    
    plot_rho_vs_T_generic(device, filename, data_column_names, 
                          plot_column_names, size=size)
    
def plot_rho_vs_T_OOP_4(size=2):
    device = OOP_4
    filename = '2021-09-26 C-Z_1.txt'
    data_column_names = ['Temperature_C', '1kHz', '10kHz', '100kHz', '1MHz']
    plot_column_names = ['10kHz', '100kHz', '1MHz']
    
    plot_rho_vs_T_generic(device, filename, data_column_names, 
                          plot_column_names, size=size, savename='_incT')
    
    filename = '2021-09-26 C-Z_2.txt'
    plot_rho_vs_T_generic(device, filename, data_column_names, 
                          plot_column_names, size=size, savename='_decT')

def main(): 
    show_all = False
    size=2
    #plot_rho_vs_T_OOP_1(size=size)
    #plot_rho_vs_T_OOP_2(size=size)
    #plot_rho_vs_T_OOP_3(size=size)
    plot_rho_vs_T_OOP_4(size=size)
    
if __name__== "__main__":
  main()
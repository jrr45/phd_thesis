# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
sys.path.append(os.path.join('..', '..', 'Code'))
import material_plotter as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

eds2_film = mp.device()
eds2_film.fileroot = 'TEM_data'
eds2_film.name = 'eds2'
eds4_film = mp.device()
eds4_film.fileroot = 'TEM_data'
eds4_film.name = 'eds4'

element_colors = [
    ('Se', '#377eb8ff'), #blue
    ('In', '#4daf4aff'), #green
    ('Au', '#ffff33ff'), #yellow
    ('Ti', '#ff7f00ff'), #orange
    ('Ga', '#984ea3ff'), #purple
    ('As', '#e41a1cff'), #red
    ('O', '#f781bfff'), #pink
    ('C', '#a65628ff'), #brown
    ]

def plot_eds_maps(film, size=2):
    for (element, color) in element_colors:
        element_file = os.path.join('TEM_data', '%s_%s.txt' % (film.name, element))
        if not os.path.exists(element_file):
            continue
        eds_data = np.genfromtxt(element_file, delimiter=",")
        
        colors_list = ["#000000ff", "#00000000"]
        colors_list.insert(1, color)
        nodes = [0.0, 0.1, 1.0]
        cmap1 = colors.LinearSegmentedColormap.from_list("mycmap",  list(zip(nodes, colors_list)))
    
    
        #eds2_se = np.nan_to_num(eds2_se)
        fig = plt.figure(figsize=(size, size), dpi=600)
        ax = fig.add_subplot(111)
        ax.imshow(eds_data, cmap=cmap1)
        plt.axis('off')
        #plt.colorbar()
        plt.show()
        mp.save_generic_svg(fig, film, '%s_%s' % (film.name, element))

def plot_HAADF_maps(film, size=2):
    element_file = os.path.join('TEM_data', '_%s_HAADF.txt' % (film.name))

    eds_data = np.genfromtxt(element_file, delimiter=",")
    
    cmap1 = plt.get_cmap('gray')

    #eds2_se = np.nan_to_num(eds2_se)
    fig = plt.figure(figsize=(size, size), dpi=600)
    ax = fig.add_subplot(111)
    ax.imshow(eds_data, cmap=cmap1)
    plt.axis('off')
    #plt.colorbar()
    plt.show()
    mp.save_generic_svg(fig, film, '_%s_HAADF' % (film.name))
    
def main():
    size = 2
    plot_eds_maps(eds2_film, size=size)
    plot_HAADF_maps(eds2_film, size=size)
    plot_eds_maps(eds4_film, size=size)
    plot_HAADF_maps(eds4_film, size=size)

if __name__== "__main__":
  main()

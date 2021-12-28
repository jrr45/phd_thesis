import sys
import os
sys.path.append(os.path.join('..', '..', 'Code'))
import material_plotter as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import hyperspy.api as hs
import scipy.ndimage

TEM_film = mp.device()
TEM_film.fileroot = 'TEM_data'

#s = hs.load(['HRETEM/4.0_1.ser'])
#s = hs.load(['/home/hoid/Nextcloud/Work/JR Thesis/Chapter-InSe-films/InSe_data/TEM_data/HRETEM/19_1.ser'])

eds3_dimensions = [91, 150, 380, 10, 90] #angle, x_start, x_end, y_start, y_end

def export_eds_data(eds_name, eds_dimensions):
    eds_data = hs.load([os.path.join(TEM_film.fileroot, eds_name + '.bcf')])
    eds_data[0].data = scipy.ndimage.rotate(eds_data[0].data, eds_dimensions[0],
                                            reshape=False)
    
    eds_data[0].crop(0, start=eds_dimensions[1], end=eds_dimensions[2])
    eds_data[0].crop(1, start=eds_dimensions[3], end=eds_dimensions[4])
    np.savetxt(os.path.join(TEM_film.fileroot,'eds3'+'_HAADF'), 
               eds_data[0].data, fmt='%.5e', delimiter=' ')
    
    eds_data[1].data = scipy.ndimage.rotate(eds_data[1].data, eds_dimensions[0],
                                            reshape=False)
    eds_data[1].crop(0, start=eds_dimensions[1], end=eds_dimensions[2])
    eds_data[1].crop(1, start=eds_dimensions[3], end=eds_dimensions[4])
    eds_data[1].set_elements(['C', 'O', 'Ti', 'Ga', 'As', 'Se', 'In', 'Au'])
    eds_data[1].set_lines(['C_Ka', 'As_Ka', 'Au_La', 'Ga_Ka', 'In_La', 'O_Ka', 'Se_Ka', 'Ti_Ka'])
    intensities = eds_data[1].get_lines_intensity()
    lines = eds_data[1].metadata.Sample.elements
    
    for (intensity, element) in zip(intensities, lines):
        np.savetxt(os.path.join(TEM_film.fileroot, 
                                eds_name + '_' + element +'.txt'), 
                   intensity.data, fmt='%.5e', delimiter=' ')

def main():
    export_eds_data('eds3', eds3_dimensions)

if __name__== "__main__":
  main()
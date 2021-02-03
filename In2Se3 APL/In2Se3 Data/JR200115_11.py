#generic imports
import os
import sys
sys.path.append(os.path.join('..','..', 'Code'))
import material_plotter as mp

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy import optimize

from matplotlib.ticker import (MultipleLocator)


fileroot = r"JR200115_11"
JR200115_11_length = 2 * 10**-6
JR200115_11_width = 3.5 * 10**-6


# R(T) for various gate voltages
RTloop_filenames = [
    'JR200115_11_034_RvsVg_300.0K.txt',
    'JR200115_11_036_RvsVg_250.0K.txt',
    'JR200115_11_037_RvsVg_200.0K.txt',
    'JR200115_11_038_RvsVg_150.0K.txt',
    'JR200115_11_039_RvsVg_100.0K.txt',
    'JR200115_11_040_RvsVg_50.0K.txt',
    'JR200115_11_041_RvsVg_10.0K.txt',
    'JR200115_11_042_RvsVg_2.0K.txt',
    ]

# IDS vs Vg loops at different T
RTloop_filenames = [
    'JR200115_11_042_RvsVg_2.0K.txt',
    'JR200115_11_041_RvsVg_10.0K.txt',
    'JR200115_11_040_RvsVg_50.0K.txt',
    'JR200115_11_039_RvsVg_100.0K.txt',
    'JR200115_11_038_RvsVg_150.0K.txt',
    'JR200115_11_037_RvsVg_200.0K.txt',
    'JR200115_11_036_RvsVg_250.0K.txt',
    'JR200115_11_034_RvsVg_300.0K.txt',
    ]

RTloop_filenames2 = [
    'JR200115_11_042_RvsVg_2.0K.txt',
    'JR200115_11_041_RvsVg_10.0K.txt',
    'JR200115_11_111_RvsVg_25.0K.txt',
    'JR200115_11_040_RvsVg_50.0K.txt',
    'JR200115_11_110_RvsVg_75.0K.txt',
    'JR200115_11_039_RvsVg_100.0K.txt',
    'JR200115_11_109_RvsVg_125.0K.txt',
    'JR200115_11_038_RvsVg_150.0K.txt',
    'JR200115_11_108_RvsVg_175.0K.txt',
    'JR200115_11_037_RvsVg_200.0K.txt',
    'JR200115_11_107_RvsVg_225.0K.txt',
    'JR200115_11_036_RvsVg_250.0K.txt',
    'JR200115_11_106_RvsVg_275.0K.txt',
    'JR200115_11_034_RvsVg_300.0K.txt',
    ]

# MR measurements, 10K has spare data
MR_sweep_files = [
    'JR200115_11_046_RvsB_75.00Vg_1.8K.txt',
    #'JR200115_11_044_RvsB_75.00Vg_2.0K.txt',
    'JR200115_11_047_RvsB_75.00Vg_5.0K.txt',
    'JR200115_11_048_RvsB_75.00Vg_10.0K.txt'
    ]

MR_sweep_files2 = [
        'JR200115_11_115_RvsB_75.00Vg_50.0K_cleaned.txt',
        'JR200115_11_117_RvsB_75.00Vg_10.0K.txt',
        'JR200115_11_113_RvsB_75.00Vg_1.8K.txt',
        ]

# 300K rate data
rate_300K_filenames = [
    'JR200115_11_060_RvsVg_300.0K.txt',
    'JR200115_11_061_RvsVg_300.0K.txt',
    'JR200115_11_062_RvsVg_300.0K.txt',
    ]

Rvst_filenames = [
    #'JR200115_11_002_Rvst_.txt',
    'JR200115_11_003_Rvst_.txt',
    #'JR200115_11_017_Rvst_.txt',
    'JR200115_11_021_Rvst_.txt',
    #'JR200115_11_035_Rvst_.txt',
    ]

IDvsVDS_loop_filenames_VG_inc = [
    'JR200115_11_121_IvsV_300K.txt',
    'JR200115_11_122_IvsV_300K.txt',
    'JR200115_11_123_IvsV_300K.txt',
    'JR200115_11_124_IvsV_300K.txt',
    'JR200115_11_125_IvsV_300K.txt',
    'JR200115_11_126_IvsV_300K.txt',
    'JR200115_11_127_IvsV_300K.txt',
    ]
IDvsVDS_loop_filenames_VG_dec = [
    'JR200115_11_134_IvsV_300K.txt',
    'JR200115_11_133_IvsV_300K.txt',
    'JR200115_11_132_IvsV_300K.txt',
    'JR200115_11_131_IvsV_300K.txt',
    'JR200115_11_130_IvsV_300K.txt',
    'JR200115_11_129_IvsV_300K.txt',
    'JR200115_11_128_IvsV_300K.txt',
    ]

IDvsVDS_one_loop_filenames_VG_inc_pos = [
    'JR200115_11_149_IvsV__300K_one_side.txt',
    'JR200115_11_150_IvsV__300K_one_side.txt',
    'JR200115_11_151_IvsV__300K_one_side.txt',
    'JR200115_11_152_IvsV__300K_one_side.txt',
    'JR200115_11_153_IvsV__300K_one_side.txt',
    'JR200115_11_154_IvsV__300K_one_side.txt',
    'JR200115_11_155_IvsV__300K_one_side.txt',
    ]

IDvsVDS_one_loop_filenames_VG_dec_pos = [
    'JR200115_11_156_IvsV__300K_one_side.txt',
    'JR200115_11_157_IvsV__300K_one_side.txt',
    'JR200115_11_158_IvsV__300K_one_side.txt',
    'JR200115_11_159_IvsV__300K_one_side.txt',
    'JR200115_11_160_IvsV__300K_one_side.txt',
    'JR200115_11_161_IvsV__300K_one_side.txt',
    'JR200115_11_162_IvsV__300K_one_side.txt',
    ]

IDvsVDS_one_loop_filenames_VG_inc_neg = [
    'JR200115_11_163_IvsV__300K_one_side.txt',
    'JR200115_11_164_IvsV__300K_one_side.txt',
    'JR200115_11_165_IvsV__300K_one_side.txt',
    'JR200115_11_166_IvsV__300K_one_side.txt',
    'JR200115_11_167_IvsV__300K_one_side.txt',
    'JR200115_11_168_IvsV__300K_one_side.txt',
    'JR200115_11_169_IvsV__300K_one_side.txt',
    ]

IDvsVDS_one_loop_filenames_VG_dec_neg = [
    'JR200115_11_170_IvsV__300K_one_side.txt',
    'JR200115_11_171_IvsV__300K_one_side.txt',
    'JR200115_11_172_IvsV__300K_one_side.txt',
    'JR200115_11_173_IvsV__300K_one_side.txt',
    'JR200115_11_174_IvsV__300K_one_side.txt',
    'JR200115_11_175_IvsV__300K_one_side.txt',
    'JR200115_11_176_IvsV__300K_one_side.txt',
    ]

IDvsVDS_5x_loop_filenames_VG_neg_inc = [
    'JR200115_11_179_IvsV__300K_one_side_5x.txt',
    'JR200115_11_180_IvsV__300K_one_side_5x.txt',
    'JR200115_11_181_IvsV__300K_one_side_5x.txt',
    'JR200115_11_182_IvsV__300K_one_side_5x.txt',
    'JR200115_11_183_IvsV__300K_one_side_5x.txt',
    'JR200115_11_184_IvsV__300K_one_side_5x.txt',
    'JR200115_11_185_IvsV__300K_one_side_5x.txt',
    ]

IDvsVDS_5x_loop_filenames_VG_pos_inc = [
    'JR200115_11_186_IvsV__300K_one_side_5x.txt',
    'JR200115_11_187_IvsV__300K_one_side_5x.txt',
    'JR200115_11_188_IvsV__300K_one_side_5x.txt',
    'JR200115_11_189_IvsV__300K_one_side_5x.txt',
    'JR200115_11_190_IvsV__300K_one_side_5x.txt',
    'JR200115_11_191_IvsV__300K_one_side_5x.txt',
    'JR200115_11_192_IvsV__300K_one_side_5x.txt',
    ]

def plot_IDSvsVg_300K_rate(log=True, size=2):
    colors = mp.colors_set1
    files = [mp.process_file(os.path.join(fileroot, x)) for x in rate_300K_filenames]
    files = mp.slice_data_each(files, 'Gate_Voltage_V', -75., -75., .1, starting_index=0)
        
    mp.plot_IDvsVg_generic(fileroot, files, "_JR200115_11_300K_rate", colors, log=log, size=size)
    


def plot_IDSvsB_custom(filenames, savename, color_order=[0,1,2,3,4,5], log=False, symm=False, size=3):    
    fig = plt.figure(figsize=(size, size), dpi=300)
    colors = mp.colors_set1
    colors = colors[color_order]
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
        
    
    ax = mp.pretty_plot_single(fig, labels=["$\it{B}$ (T)", '$\it{ΔR_{DS}/R_{DS}(0)}$ (%)'],
                             yscale=('log' if log else 'linear'))
    
    for (file, color) in zip(files, colors):
        xval = file['Magnetic_Field_T']
        yval = file['Resistance_1_Ohms']
        
        tolerance = .001
        #yval2 = yval[mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0):]
        yval = yval[:mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0)]
        xval = xval[:mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0)]
        
        print(np.size(yval))
        bin_size = 3
        R = yval
        R = R.reshape((-1, bin_size))
        R = np.mean(R, axis=1)
        H = xval
        H = H.reshape((-1, bin_size))
        H = np.mean(H, axis=1)
        
        if symm:
             R = (R+R[::-1])/2
        
        #print(H)
        y0 = R[np.argmin(np.abs(H))] 
        
        # % deviation from R(H=0)
        R = 100*(R-y0)/y0
        
        #filter high
        ind = np.where(np.abs(H) < 9)
        H = H[ind]
        R = R[ind]

        #Remove holes
        ind = np.where(np.isfinite(R))
        H = H[ind]
        R = R[ind]
        
        ax.plot(H, R, '-', ms=3, linewidth=1., color=color)
        
        ind = np.where(H > 0)
        H = H[ind]
        R = R[ind]
        popt, pcov = optimize.curve_fit(WL, H, R)
        Rfit = WL(H, popt[0], popt[1])
        ax.plot(H, Rfit, '-', ms=3, linewidth=1., color='black')
    
    ax.set_ylim((-1.2, .5))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    
    mp.save_generic_svg(fig, fileroot, savename)


def F(Bx, B):
    return np.log(Bx/B) - special.digamma(.5 + (Bx/B))
def F2(Bx, B):
    return np.log(B/Bx) + special.digamma(.5 + 1./(B/Bx))
def F3(Bx, B):
    return np.log(Bx/B) + special.digamma(.5 + 1./(Bx/B))
def WL_wiki2(B, Bx):
    (Bϕ, BSO, Be) = (Bx[0], Bx[1], B[2])
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
def WL_wiki(B, Bϕ, BSO, Be):
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
def WL(B, Bϕ, BSO, Be, α):
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
    ##return -e22π2ħ * (F2(BSO + Bϕ, B) - .5*(F2(Bϕ, B) - F2(2*BSO + Bϕ, B))) + α*B**2 # TaSe
    ##return -e22π2ħ * (.5*F(Bϕ, B) - F(Bϕ+BSO, B) - .5*F(2*BSO + Bϕ, B)) #VSe2
    #return α*e22π2ħ*F(Bϕ, B) # SSOC


def plot_σvsB_custom(filenames, savename, color_order=[0,1,2,3,4,5], log=False, \
                     fit_lim=np.inf, size=2, fontsize=10, labelsize=10, xmult=4):    
    #from curve_fit import annealing
    fig = plt.figure(figsize=(size, size), dpi=300)
    colors = mp.colors_set1
    colors = colors[color_order]
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in filenames]
        
    ax = mp.pretty_plot_single(fig, labels=["$\\it{B}$ (T)", '$\it{Δσ}\,  (nS)}$'],
                             yscale=('log' if log else 'linear'), fontsize=fontsize, labelsize=labelsize)
    
    for (file, color) in zip(files, colors):
        xval = file['Magnetic_Field_T']
        yval = file['Resistance_1_Ohms']
        Temperature_K = file['Temperature_K'][0]
        
        tolerance = .001
        #yval2 = yval[mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0):]
        yval = yval[:mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0)]
        xval = xval[:mp.first_occurance_1D(xval, 9, tol=tolerance, starting_index=0)]
        
        # average over a given bin size
        bin_size = 3
        R = yval
        R = R.reshape((-1, bin_size))
        R = np.mean(R, axis=1)
        H = xval
        H = H.reshape((-1, bin_size))
        H = np.mean(H, axis=1)
        
        #symmetrize it
        R = (R+R[::-1])/2
        
        Rsq = mp.sheet_resistance(R, JR200115_11_length, JR200115_11_width)
        
        #print(H)
        Rsq_0 = Rsq[np.argmin(np.abs(H))] 
        
        # deviation from σ(H=0)
        Δσ = -(Rsq-Rsq_0)/(Rsq_0**2)
        #Δσ2 = (1./Rsq-1./Rsq_0)
        
        e2π2ħ = (mp.fundamental_charge_e**2)/(np.pi**2 * mp.ħ) 
        
        #Remove holes
        ind = np.where(np.isfinite(Δσ))
        H = H[ind]
        Δσ = Δσ[ind]
        
        print(e2π2ħ)

        ax.plot(H, Δσ * 10**9, '-', ms=3, linewidth=1., color=color)
        
        #fit stuff
        if Temperature_K > fit_lim:
            continue
        
        ind = np.where(H > 0)
        H_half = H[ind]
        Δσ  = Δσ[ind]
                
        bounds = ([0,0,0],1000)
        #bounds = [0,1000],[0,1000],[0,1000]]
        best_r2 = 0
        best_popt = [1,1,1]
        
        
        for i in range(1,10):
            #result = annealing.curve_fit(WL_wiki2, H, Δσ, bounds=bounds)

            #popt = result.x # optimal fit parameters
            
            p0 = np.ndarray.flatten(np.random.rand(1,3))
            #p0[3] = p0[3] - .5
            p0 = np.multiply(p0,.0001*i)
            #p0 = [0.20277506744303325, 7.959480841631488e-07, 0.2043696078628539]
            
            popt, pcov = optimize.curve_fit(WL_wiki, H_half, Δσ, maxfev=10000, ftol=10**-15, \
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
        Δσfit = WL_wiki(abs(H), *best_popt)
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
    ax.xaxis.set_major_locator(MultipleLocator(xmult))
    ax.set_ylim((-5,105))

    mp.save_generic_svg(fig, fileroot, savename)
    plt.show()
    plt.clf()
    
def plot_IDvsVDS_loops(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[0,4,3,6,2,8,1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_loop_filenames_VG_inc]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)
    mp.plot_IDvsVDS_generic(fileroot, files, 'VDS_loop_inc', colors, log=log, \
                              size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                              ylim=ylim)
        
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_loop_filenames_VG_dec]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)
    mp.plot_IDvsVDS_generic(fileroot, files, 'VDS_loop_dec', colors, log=log, \
                              size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                              ylim=ylim)

def plot_IDvsVDS_loops2(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[0,4,3,6,2,8,1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    subset = [0,1,2,3,4,5,6]
    colors = [colors[i] for i in subset]
    IDvsVDS_one_loop_filenames_VG_inc_posss = [IDvsVDS_one_loop_filenames_VG_inc_pos[i] for i in subset]
    IDvsVDS_one_loop_filenames_VG_inc_negss = [IDvsVDS_one_loop_filenames_VG_inc_neg[i] for i in subset]
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_one_loop_filenames_VG_inc_posss]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=1, nth_finish=2)
    mp.plot_IDvsVDS_generic(fileroot, files, 'VDS_one_loop_inc_pos', colors, log=log, \
                              figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                              ylim=ylim)
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_one_loop_filenames_VG_inc_negss]
    #print(files[0]['Gate_Voltage_V'])
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=1, nth_finish=2)
    mp.plot_IDvsVDS_generic(fileroot, files, 'VDS_one_loop_inc_neg', colors, log=log, \
                              figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                              ylim=ylim)
    

def plot_IDvsVDS_5x_loops(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in ['JR200115_11_177_IvsV__300K_one_side.txt']]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 10., .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name = str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(fileroot, [file], 'VDS_5x_loop_neg_one' + add_name, colors, log=log, \
                                  figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in ['JR200115_11_178_IvsV__300K_one_side.txt']]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, -10, .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name =  str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(fileroot, [file], 'VDS_5x_loop_pos_one' + add_name, colors, log=log, \
                                  figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    

def plot_IDvsVDS_5x_loops2(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_5x_loop_filenames_VG_neg_inc]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0., .01, nth_start=1, nth_finish=5)
    for file in files:
        add_name = str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(fileroot, [file], 'VDS_5x_loop_neg_' + add_name, colors, log=log, \
                                  figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    return
    files = [mp.process_file(os.path.join(fileroot, x)) for x in IDvsVDS_5x_loop_filenames_VG_pos_inc]
    #files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)
    for file in files:
        add_name =  str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(fileroot, [file], 'VDS_5x_loop_pos_' + add_name, colors, log=log, \
                                  figsize=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    
def main(): #sample A
    show_all = False
    # -- Plot ID vs VG loops
    if False or show_all:
        mp.plot_IDvsVg_each(fileroot, RTloop_filenames, '_JR200115_11', log=True, size=2, majorx=40,
                          ylim=(10**-10,None), fontsize=10, labelsize=10)
    
    # -- Cross section of loop data
    if False or show_all:
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR200115_11", figsize=1.5, ylim=(0, 1.5), log=False)
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR200115_11_inset", figsize=.7, \
                                      log=False, increments=[75], fontsize=10, labelsize=10, colororder=[1])
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR200115_11", figsize=2,\
                                      xlim=(None, None), ylim=(None, None), log=True, increments=[0, 25, 50, 75], colororder=[0, 3,2,1])
        mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR200115_11", figsize=2,\
                                      xlim=(None, None), ylim=(None, None), log=False, increments=[25, 50, 75], colororder=[3,2,1])
        #mp.plot_loopR_cross_section(fileroot, RTloop_filenames, "_JR200115_11", figsize=2, ylim=(None, None),\
        #                              log=True, increments=[0, 25, 50, 75], colororder=[0,3,2,1])
    
    # -- 300K ID vs VDS curves
    if False or show_all:
        mp.plot_IDvVDS_gating_generic(fileroot, 'JR200115_11_', '_IvsV_300.0K.txt', 64, 7, "_300K", \
                                        figsize=2, xadj=0, log=False, majorx=1)
        mp.plot_IDvVDS_gating_generic(fileroot, 'JR200115_11_', '_IvsV_300.0K.txt', 64, 7, "_300K", \
                                        figsize=2, xadj=0, log=True, majorx=1)
    
    # -- 300K rate
    if False:
        plot_IDSvsVg_300K_rate()
        
        
    # -- size of loop
    if False:
        mp.plot_ΔVGvT(fileroot, RTloop_filenames, 10**-8, size=2)
    
    
    if False:
        files = [mp.process_file(os.path.join(fileroot, x)) for x in Rvst_filenames]    
        mp.plot_IDSvsTime_generic(fileroot, files, '_RvsTime', log=False, size=2, majorx=1800, ylim=(None,None))
    
        
    # still working on    
    if False or show_all:
        plot_σvsB_custom(MR_sweep_files2[1:], '_JR200115_11_MR_sweep_1.8', color_order=[1,0],\
                         fit_lim=10, size=2)
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep_both', color_order=[0,1],\
                         fit_lim=60, size=2)
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep', color_order=[0,1],\
                         fit_lim=0, size=2)
            
    # -- carrier mobility μ
    if False or show_all:
        mp.plot_mobility_μ_cross_section(fileroot, RTloop_filenames, "_JR200115_11", JR200115_11_length, JR200115_11_width, figsize=1.5, ylim=(None, None),\
                                           log=False, increments=[25, 50, 75], colororder=[3,2,1])
    
    # loops if ID vs VDS showing hysteresis
    if False  or show_all:
        plot_IDvsVDS_loops(log=True,figsize=2, fontsize=10, labelsize=10)
        plot_IDvsVDS_loops(log=False,figsize=2, fontsize=10, labelsize=10)
        #plot_IDvsVDS_loops2(log=False,figsize=2)
        #plot_IDvsVDS_5x_loops(figsize=2, fontsize=10, labelsize=8, log=False)
    
    if True or show_all:
        plot_σvsB_custom(MR_sweep_files2[2:], '_JR200115_11_MR_sweep_1.8', color_order=[1,0],\
                            fit_lim=10, size=2,fontsize=10, labelsize=10, xmult=4)
        
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep_fit', color_order=[0,2,1],\
                            fit_lim=100, size=2,fontsize=10, labelsize=10, xmult=4)
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep', color_order=[0,2,1],\
                            fit_lim=1, size=2,fontsize=10, labelsize=10, xmult=4)
    
if __name__== "__main__":
  main()

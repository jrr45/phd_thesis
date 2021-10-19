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

JR200115_11 = mp.flake_device()
fileroot = r"JR200115_11"
JR200115_11.fileroot = "JR200115_11"
JR200115_11.name = "JR200115_11"
JR200115_11.length = 2 * 10**-6
JR200115_11.width = 3.5 * 10**-6
JR200115_11.thickness = 20 * 10**-9


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
    
    mp.save_generic_svg(fig, JR200115_11, savename)

# Reuse WL expression
def F(Bx, B):
    return np.log(Bx/B) - special.digamma(.5 + (Bx/B))
def F2(Bx, B):
    return np.log(B/Bx) + special.digamma(.5 + 1./(B/Bx))
def F3(Bx, B):
    return np.log(Bx/B) + special.digamma(.5 + 1./(Bx/B))

#pack as vector
def WL_wiki2(B, Bx):
    (Bϕ, BSO, Be) = (Bx[0], Bx[1], B[2])
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
# full WL formula
def WL_wiki(B, Bϕ, BSO, Be):
    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
#def WL(B, Bϕ, BSO, Be, α):
#    e22π2ħ = (mp.fundamental_charge_e**2)/(2*np.pi**2 * mp.ħ) 
#    return e22π2ħ * (.5*F(Bϕ, B) + F(BSO+Be, B) - 3*.5*F(3*.5*BSO + Bϕ, B))  #WIKI
    ##return -e22π2ħ * (F2(BSO + Bϕ, B) - .5*(F2(Bϕ, B) - F2(2*BSO + Bϕ, B))) + α*B**2 # TaSe
    ##return -e22π2ħ * (.5*F(Bϕ, B) - F(Bϕ+BSO, B) - .5*F(2*BSO + Bϕ, B)) #VSe2
    #return α*e22π2ħ*F(Bϕ, B) # SSOC


def plot_σvsB_custom(filenames, savename, color_order=[0,1,2,3,4,5], log=False, \
                     fit_lim=np.inf, size=2, fontsize=10, labelsize=10, xmult=4):    
    #from curve_fit import annealing
    fig = plt.figure(figsize=(size, size), dpi=300)
    colors = mp.colors_set1
    colors = colors[color_order]
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in filenames]
        
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
        
        Rsq = mp.sheet_resistance(JR200115_11, R)
        
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
        
        # run multiple times and take the best one
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
        Δσfit = WL_wiki(H, *best_popt)
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

    mp.save_generic_svg(fig, JR200115_11, savename)
    plt.show()
    plt.clf()

def get_IDvsVDS_loop_files():
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in IDvsVDS_loop_filenames_VG_inc]
    files_loop_inc = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)

    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in IDvsVDS_loop_filenames_VG_dec]
    files_loop_dec = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)

    return (files_loop_inc, files_loop_dec)

def get_IDvsVDS_loop_files_side():
    subset = [0,1,2,3,4,5,6]
    IDvsVDS_one_loop_filenames_VG_pos_inc = [IDvsVDS_one_loop_filenames_VG_inc_pos[i] for i in subset]
    IDvsVDS_one_loop_filenames_VG_pos_dec = [IDvsVDS_one_loop_filenames_VG_dec_pos[i] for i in subset]
    IDvsVDS_one_loop_filenames_VG_neg_inc = [IDvsVDS_one_loop_filenames_VG_inc_neg[i] for i in subset]
    IDvsVDS_one_loop_filenames_VG_neg_dec = [IDvsVDS_one_loop_filenames_VG_dec_neg[i] for i in subset]



    files_loop_pos_inc = [mp.process_file(os.path.join(JR200115_11.fileroot, x))
                          for x in IDvsVDS_one_loop_filenames_VG_pos_inc]
    files_loop_pos_dec = [mp.process_file(os.path.join(JR200115_11.fileroot, x))
                          for x in IDvsVDS_one_loop_filenames_VG_pos_dec]
    files_loop_neg_inc = [mp.process_file(os.path.join(JR200115_11.fileroot, x))
                          for x in IDvsVDS_one_loop_filenames_VG_neg_inc]
    files_loop_neg_dec = [mp.process_file(os.path.join(JR200115_11.fileroot, x))
                          for x in IDvsVDS_one_loop_filenames_VG_neg_dec]

    return (files_loop_pos_inc, files_loop_pos_dec, files_loop_neg_inc, files_loop_neg_dec)
 

def plot_2xloops_IDvsVDS(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.get_IDvsVDS_colors()
    #colors = mp.colors_set1[[0,4,0,2,2,8,1]]
    (files_loop_pos_inc, files_loop_pos_dec, files_loop_neg_inc, files_loop_neg_dec) \
        = get_IDvsVDS_loop_files_side()

    savename = "JR200115_11_2xloops_"
    device = JR200115_11

    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)

    mp.plot_IDvsVDS_generic(device, files_loop_pos_inc[0:7:3], 'VDS_2xloops_pos_inc', 
                            colors[0:7:3], log=log, size=figsize, xadj=0, x_mult=5,
                            fontsize=fontsize, labelsize=labelsize, ylim=ylim)
    
    mp.plot_IDvsVDS_generic(device, files_loop_pos_dec[0:7:3], 'VDS_2xloops_pos_dec', 
                            reversed(colors[0:7:3]), log=log, size=figsize, xadj=0, x_mult=5,
                            fontsize=fontsize, labelsize=labelsize, ylim=ylim)
    
    mp.plot_IDvsVDS_generic(device, files_loop_neg_inc[0:7:3], 'VDS_2xloops_neg_inc', 
                            colors[0:7:3], log=log, size=figsize, xadj=0, x_mult=5,
                            fontsize=fontsize, labelsize=labelsize, ylim=ylim)
    
    mp.plot_IDvsVDS_generic(device, files_loop_neg_dec[0:7:3], 'VDS_2xloop_neg_dec', 
                            reversed(colors[0:7:3]), log=log, size=figsize, xadj=0, x_mult=5,
                            fontsize=fontsize, labelsize=labelsize, ylim=ylim)
   
def plot_loops_IDvsVDS(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.get_IDvsVDS_colors()
    (files_loop_inc, files_loop_dec) = get_IDvsVDS_loop_files()

    savename = "JR200115_11_loops_"
    device = JR200115_11

    colors = mp.colors_set1[[0,4,3,6,2,8,1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    mp.plot_IDvsVDS_generic(device, files_loop_inc, 'VDS_loop_inc', colors, log=log,
                            size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                            ylim=ylim)
    
    mp.plot_IDvsVDS_generic(device, files_loop_dec, 'VDS_loop_dec', colors, log=log,
                            size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                            ylim=ylim)
    

def plot_IDvsVDS_2x_loops(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in ['JR200115_11_177_IvsV__300K_one_side.txt']]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 10., .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name = str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_neg_one' + add_name, colors, log=log, \
                                  size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in ['JR200115_11_154_IvsV__300K_one_side.txt']]
    #files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name =  str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_pos_one' + add_name, colors, log=log, \
                                  size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)

def plot_IDvsVDS_5x_loops(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in ['JR200115_11_177_IvsV__300K_one_side.txt']]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 10., .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name = str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_neg_one' + add_name, colors, log=log, \
                                  size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in ['JR200115_11_154_IvsV__300K_one_side.txt']]
    #files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=1, nth_finish=4)
    for file in files:
        add_name =  str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_pos_one' + add_name, colors, log=log, \
                                  size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                  ylim=ylim)
    

def plot_IDvsVDS_5x_loops2(figsize=1.5, fontsize=10, labelsize=8, log=True):
    colors = mp.colors_set1[[1]]
    if not log:
        ylim = None
    else:
        ylim = (10**-12, 2.1*10**-7)
    
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in IDvsVDS_5x_loop_filenames_VG_neg_inc]
    files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0., .01, nth_start=1, nth_finish=5)
    for file in files:
        add_name = str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_neg_' + add_name, colors, log=log, \
                                size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                ylim=ylim)
    return
    files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in IDvsVDS_5x_loop_filenames_VG_pos_inc]
    #files = mp.slice_data_each(files, 'Voltage_1_V', 0, 0, .01, nth_start=2, nth_finish=2)
    for file in files:
        add_name =  str(round(file['Gate_Voltage_V'][0],1)).zfill(3) + 'V'
        
        mp.plot_IDvsVDS_generic(JR200115_11, [file], 'VDS_5x_loop_pos_' + add_name, colors, log=log, \
                                size=figsize, xadj=0, x_mult=5, fontsize=fontsize, labelsize=labelsize,
                                ylim=ylim)

def plot_loops_IDVvsVDS_generic(fitfunciton, figsize=2, fitR2=.996, fitpoints=10):
    colors = mp.get_IDvsVDS_colors()
    (files_loop_inc, files_loop_dec) = get_IDvsVDS_loop_files()
    files_loop_inc = mp.slice_data_each(files_loop_inc, 'Voltage_1_V', .1, .1, .01, nth_start=1, nth_finish=1)
    files_loop_dec = mp.slice_data_each(files_loop_dec, 'Voltage_1_V', .1, .1, .01, nth_start=1, nth_finish=1)

    
    savename = 'JR200115_11_loops_'
    device = JR200115_11
    
    fitfunciton(device, files_loop_inc, savename + '_IDvVDS-positive_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_loop_dec, savename + '_IDvVDS-positive_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_loops_IDVvsVDS_Play(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDVvsVDS_Play_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_loops_IDVvsVDS_Thermionic(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDVvsVDS_Thermionic_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_loops_IDVvsVDS_DirectTunneling(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDVvsVDS_DirectTunneling_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_loops_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDVvsVDS_FowlerNordheim_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_loops_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDVvsVDS_PooleFrenkel_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_loops_IDvsVD_Schottky(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDvsVD_Schottky_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_loops_IDvsVDS_SCLC(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDvsVDS_SCLC_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_loops_IDvsVDS_power(figsize=2, fitR2=.996, fitpoints=10):
    plot_loops_IDVvsVDS_generic(mp.plot_IDvsVDS_power_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)
    

def get_300K_IDvsVDS_files():
    basename = 'JR200115_11_'
    
    start = 64
    filenames = [(basename + str(i).zfill(3) + '_IvsV_300.0K.txt') for i in range(start, start+7)]
    files_pos_inc = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i).zfill(3) + '_IvsV_300.0K.txt') for i in range(start+7, start+14)]
    files_pos_dec = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i).zfill(3) + '_IvsV_300.0K.txt') for i in range(start+14, start+21)]
    files_neg_inc = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in filenames]
    
    filenames = [(basename + str(i).zfill(3) + '_IvsV_300.0K.txt') for i in range(start+21, start+28)]
    files_neg_dec = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in filenames]
    
    return (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec)

# 300K IV plots
def plot_300K_IDvsVDS(figsize=1.5, log=False):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_300K_IDvsVDS_files()

    savename = "JR200115_11_300K_"
    device = JR200115_11
    xadj = 1
    
    mp.plot_IDvsVDS_generic(device, files_pos_inc, savename + '_IDvVDS-positive_VG-increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_pos_dec, savename + '_IDvVDS-positive_VG-decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_inc, savename + '_IDvVDS-negative_VG-increasing', 
                            colors, size=figsize, xadj=xadj, log=log)
    
    mp.plot_IDvsVDS_generic(device, files_neg_dec, savename + '_IDvVDS-negative_VG-decreasing', 
                            reversed(colors), size=figsize, xadj=xadj, log=log)
    
def plot_300K_IDVvsVDS_generic(fitfunciton, figsize=2, fitR2=.996, fitpoints=10):
    colors = mp.get_IDvsVDS_colors()
    (files_pos_inc, files_pos_dec, files_neg_inc, files_neg_dec) = get_300K_IDvsVDS_files()
    
    savename = 'JR200115_11_300K_'
    device = JR200115_11
    
    fitfunciton(device, files_pos_inc, savename + '_IDvVDS-positive_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_pos_dec, savename + '_IDvVDS-positive_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_inc, savename + '_IDvVDS-negative_VG-increasing', 
                            colors, size=figsize, fitR2=fitR2, fitpoints=fitpoints)
    
    fitfunciton(device, files_neg_dec, savename + '_IDvVDS-negative_VG-decreasing', 
                            reversed(colors), size=figsize, fitR2=fitR2, fitpoints=fitpoints) 

def plot_300K_IDVvsVDS_Play(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_Play_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_300K_IDVvsVDS_Thermionic(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_Thermionic_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_300K_IDVvsVDS_DirectTunneling(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_DirectTunneling_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_300K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_FowlerNordheim_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
    
def plot_300K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDVvsVDS_PooleFrenkel_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_300K_IDvsVD_Schottky(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVD_Schottky_generic,
                           figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)    
        
def plot_300K_IDvsVDS_SCLC(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVDS_SCLC_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)

def plot_300K_IDvsVDS_power(figsize=2, fitR2=.996, fitpoints=10):
    plot_300K_IDVvsVDS_generic(mp.plot_IDvsVDS_power_generic,
                               figsize=figsize, fitR2=fitR2, fitpoints=fitpoints)


def main(): #sample A
    show_all = False
    # -- Plot ID vs VG loops
    if False or show_all:
        mp.plot_IDvsVg_each(JR200115_11, RTloop_filenames, '_JR200115_11', log=True, size=2, majorx=40,
                          ylim=(10**-10,None), fontsize=10, labelsize=10)
    
    # -- Cross section of loop data
    if False or show_all:
        mp.plot_loopR_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", figsize=1.5, xlim=(0,322), ylim=(0, 1.5), log=False)
        mp.plot_loopR_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11_inset", figsize=.7, \
                                      log=False, increments=[75], fontsize=10, labelsize=10, colororder=[1], xlim=(0,322))
        mp.plot_loopR_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", figsize=2,\
                                       xlim=(0,322), ylim=(None, None), log=True, increments=[0, 25, 50, 75], colororder=[0, 3,2,1])
        mp.plot_loopR_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", figsize=2,\
                                       xlim=(0,322), ylim=(None, None), log=False, increments=[25, 50, 75], colororder=[3,2,1])
        mp.plot_loopR_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11_more", figsize=2, ylim=(None, None),\
                                      log=True, increments=[-15, -10, -5, 0, 25, 50, 75], colororder=[4,5,6, 0,3,2,1])
    
    # -- 300K ID vs VDS curves
    if False or show_all:
        plot_300K_IDvsVDS(figsize=2, log=True)
        plot_300K_IDvsVDS(figsize=2, log=False)
    
    # -- 300K rate
    if False:
        plot_IDSvsVg_300K_rate()
        
        
    # -- size of loop
    if False:
        mp.plot_ΔVGvT(JR200115_11, RTloop_filenames, 10**-8, size=2)
    
    
    if False:
        files = [mp.process_file(os.path.join(JR200115_11.fileroot, x)) for x in Rvst_filenames]    
        mp.plot_IDSvsTime_generic(JR200115_11, files, '_RvsTime', log=False, size=2, majorx=1800, ylim=(None,None))
    
            
    # -- carrier mobility μ
    if False or show_all:
        mp.plot_mobility_μ_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11",
                                         figsize=1.5, ylim=(None, None),\
                                         log=False, increments=[25, 50, 75], colororder=[3,2,1])
    
    # loops if ID vs VDS showing hysteresis
    if False  or show_all:
        plot_loops_IDvsVDS(log=True,figsize=2, fontsize=10, labelsize=10)
        plot_loops_IDvsVDS(log=False,figsize=2, fontsize=10, labelsize=10)
        #plot_loops_IDvsVDS2(log=False,figsize=2)
        #plot_IDvsVDS_5x_loops(figsize=2, fontsize=10, labelsize=8, log=False)
        #plot_IDvsVDS_5x_loops2(figsize=2, fontsize=10, labelsize=8, log=False)
        plot_2xloops_IDvsVDS(figsize=2, fontsize=10, labelsize=8, log=False)
        
    if False or show_all:
        fitR2 = .99
        fitpoints=10
        plot_loops_IDvsVDS_power(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDvsVDS_SCLC(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDvsVD_Schottky(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDVvsVDS_DirectTunneling(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDVvsVDS_Thermionic(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_loops_IDVvsVDS_Play(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
    
    if False or show_all:
        plot_σvsB_custom(MR_sweep_files2[2:], '_JR200115_11_MR_sweep_1.8', color_order=[1,0],\
                            fit_lim=10, size=2,fontsize=10, labelsize=10, xmult=4)
        
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep_fit', color_order=[0,2,1],\
                            fit_lim=100, size=2,fontsize=10, labelsize=10, xmult=4)
        plot_σvsB_custom(MR_sweep_files2, '_JR200115_11_MR_sweep', color_order=[0,2,1],\
                            fit_lim=1, size=2,fontsize=10, labelsize=10, xmult=4)

    # min subthreshold slope
    if False or show_all:
        mp.plot_maxSS_vs_T(JR200115_11, RTloop_filenames, '_minSSvsT', Npoints=5, Icutoff=10*10**-11)
    
    # delta voltage threshold
    if False or show_all:
        mp.plot_ΔVTvT(JR200115_11, RTloop_filenames, '_ΔVTvsT', Npoints=5, Icutoff=10*10**-11)
    
    # Schotty and Schotty-Simmons for fixed V. ln(J*T^pow) vs 1/T
    if False or show_all:
        mp.plot_Schottky_Simmons_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", 
                figsize=2, xlim=(0,None),
                ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])
        mp.plot_Schottky_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", 
                figsize=2, xlim=(0,None),
                ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])
        
    #mp.plot_play_cross_section(JR200115_11, RTloop_filenames, "_JR200115_11", 
    #            figsize=2, xlim=(0,None),
    #            ylim=(None, None), increments=[0, 25, 50, 75], colororder=[0,3,2,1])

    # fit ID vs VDS data    
    if True or show_all:
        fitR2 = .99
        fitpoints=3
        plot_300K_IDvsVDS_power(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDvsVDS_SCLC(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDvsVD_Schottky(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_PooleFrenkel(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_FowlerNordheim(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_DirectTunneling(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_Thermionic(figsize=2, fitR2=fitR2, fitpoints=fitpoints)
        #plot_300K_IDVvsVDS_Play(figsize=2, fitR2=fitR2, fitpoints=fitpoints)

if __name__== "__main__":
  main()

import Jpython_plotter as jpp
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

datafileroot = '.'
savefileroot = 'Generated Figures'

def plot_landau_1st_PvT(figsize=(2, 2)):
    colors = jpp.colors_set1
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$", '$\it{P}$'],
                             yscale='linear')
    
    
    (F0, α0, β, γ)  = (0, 1, -5, .5) # F0, alpha, beta, gamma, E
    T0 = 100
    TC = T0 + 3*β*β/(16*α0*γ)
    T1 = T0 + β*β/(4*α0*γ)
    print("TC: %f; T1: %f" % (TC,T1))
    
    Temperature = np.linspace(0, 250, 100, endpoint=True)
    
    Efields = [0,0,0,0,0]
    P0 = np.zeros(np.size(Temperature))
    
    Efields = [0, 15, 30, 100]
    
    for (Efield, color) in zip(Efields, colors):
        Ps = []
        
        
        for T in Temperature:
            def FEnergy(P):
                return (F0
                + α0*(T-T0)*np.power(P,2)/2 
                + β*np.power(P,4)/4
                + γ*np.power(P,6)/6
                - Efield*P
                )
            
            res = optimize.minimize(FEnergy, [10], method="L-BFGS-B")
            Ps.append(res.x[0])
            
        
        ax.plot(Temperature, np.array(Ps), '-', ms=3, linewidth=1.5, color=color)
    
    plt.axis('off')
    
    jpp.save_generic_svg(fig, savefileroot, "landau_1st_PvT")
    
def plot_landau_2nd_PvT(figsize=(2, 2)):
    colors = jpp.colors_set1
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{T}$", '$\it{P}$'],
                             yscale='linear')
    
    
    (F0, α0, β, γ)  = (0, 1, 5, .1) # F0, alpha, beta, gamma
    T0 = 100
    TC = T0 + 3*β*β/(16*α0*γ)
    T1 = T0 + β*β/(4*α0*γ)
    print("TC: %f; T1: %f" % (TC,T1))
    
    Temperature = np.linspace(0, 250, 100, endpoint=True)
    
    Efields = [0,0,0,0,0]
    P0 = np.zeros(np.size(Temperature))
    
    Efields = [0,30]
    
    for (Efield, color) in zip(Efields, colors):
        Ps = []
        
        
        for T in Temperature:
            def FEnergy(P):
                return (F0
                + α0*(T-T0)*np.power(P,2)/2 
                + β*np.power(P,4)/4
                + γ*np.power(P,6)/6
                - Efield*P
                )
            
            res = optimize.minimize(FEnergy, [10], method="L-BFGS-B")
            Ps.append(res.x[0])
            
        
        ax.plot(Temperature, np.array(Ps), '-', ms=3, linewidth=1.5, color=color)
    
    plt.axis('off')
    
    jpp.save_generic_svg(fig, savefileroot, "landau_2nd_PvT")
    
def plot_landau_1st_FvP(figsize=(2, 2)):
    colors = jpp.colors_set1
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{P}$", '$\it{F}$'],
                             yscale='linear')
    
    
    (F0, α0, β, γ)  = (0, 1, -5, .5) # F0, alpha, beta, gamma, E
    T0 = 100
    TC = T0 + 3*β*β/(16*α0*γ)
    T1 = T0 + β*β/(4*α0*γ)
    print("TC: %f; T1: %f" % (TC,T1))
    temperatures = [ 
        150, T1, TC, T0, 90
    ]
    Efields = [0,0,0,0,0]
    
    Fmax = []
    Fmin = []
    
    for (T, Efield, color) in zip(temperatures, Efields, colors):
        Pmax = 5
        points = 100
        Polarization = np.linspace(-Pmax, Pmax, points, endpoint=True)
        FEnergy = (F0
                   + α0*(T-T0)*np.power(Polarization,2)/2 
                   + β*np.power(Polarization,4)/4
                   + γ*np.power(Polarization,6)/6
                   - Efield*Polarization
                   )
        
        split = np.int(np.round(points/2,0))
        Fmax.append(np.max(FEnergy[:split]))
        Fmax.append(np.max(FEnergy[split:]))
        Fmin.append(np.min(FEnergy[:split]))
        Fmin.append(np.min(FEnergy[split:]))
        
        ax.plot(Polarization, FEnergy, '-', ms=3, linewidth=1.5, color=color)
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    Fmin = np.min(Fmin)
    ax.set_ylim((Fmin-np.abs(Fmin)*.1, np.min(Fmax)))
    
    plt.axis('off')
    
    jpp.save_generic_svg(fig, savefileroot, "landau_1st_FvP")
    
def plot_landau_2nd_FvP(figsize=(2, 2)):
    colors = jpp.colors_set1
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = jpp.pretty_plot_single(fig, labels=["$\it{P}$", '$\it{F}$'],
                             yscale='linear')
    
    
    (F0, α0, β, γ)  = (0, 1, 5, .1) # F0, alpha, beta, gamma
    T0 = 100
    temperatures = [ 
        150, T0, 50, 50
    ]
    Efields = [0,0,0,20]
    
    Fmax = []
    Fmin = []
    
    for (T, Efield, color) in zip(temperatures, Efields, colors):
        Pmax = 5
        points = 100
        Polarization = np.linspace(-Pmax, Pmax, points, endpoint=True)
        FEnergy = (F0
                   + α0*(T-T0)*np.power(Polarization,2)/2 
                   + β*np.power(Polarization,4)/4
                   + γ*np.power(Polarization,6)/6
                   - Efield*Polarization
                   )
        
        split = np.int(np.round(points/2,0))
        Fmax.append(np.max(FEnergy[:split]))
        Fmax.append(np.max(FEnergy[split:]))
        Fmin.append(np.min(FEnergy[:split]))
        Fmin.append(np.min(FEnergy[split:]))
        
        ax.plot(Polarization, FEnergy, '-', ms=3, linewidth=1.5, color=color)
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    Fmin = np.min(Fmin)
    ax.set_ylim((Fmin-np.abs(Fmin)*.1, np.min(Fmax)))
    
    plt.axis('off')
    
    jpp.save_generic_svg(fig, savefileroot, "landau_2nd_FvP")


def main():
    plot_landau_1st_FvP(figsize=(2, 2))
    plot_landau_1st_PvT(figsize=(2, 2))
    plot_landau_2nd_FvP(figsize=(2, 2))  
    plot_landau_2nd_PvT(figsize=(2, 2))

if __name__== "__main__":
  main()
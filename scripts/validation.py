# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:16:52 2020
This script does the following:
    1. performs model validation on the infiltration model (Govindarju et. al (2001) and Goyal et. al (2019))
    2. creates subplots to compare the observed and simulated infiltration values (uses colorplots.py module)

INPUTS:
    1. The 95% CR selected parameters (muy and sigy) as numpy arrays (created by the conflation.py script)
    2. The rainfall-runoff experiments data from ../dataset/experiment_data/' directory
    
OUTPUTS:
    1. A list 'I' containing the [time, Iobs, Imodel_max, Imodel_min, Imodel_mostliekly] for all the validation experiments
    2. subplots comparing the Iobs and Imodel, using "val_plot" library in the "colorplots" module.
    
@author: abhishek
"""

# %% Validation
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import fsolve
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from infiltration_model import cuminfil, arealinfil
# from colorplots import val_plot




# kwarg = 'GP_1'
# kwarg = 'linpool'

inst = 'dri'
varfile = '../variables/coarsening/'      # location of the numpy variables
# musel = np.load(varfile + 'musel_%s.npy' %kwarg)
# sigsel = np.load(varfile + 'sigsel_%s.npy' %kwarg)

musel = np.load(varfile + 'musel_' + inst + '_95.npy')
sigsel = np.load(varfile + 'sigsel_' + inst + '_95.npy')


# % User inputs
fileloc = r'../dataset/experiment_data/'
# exp_no = 8             # Experiment number
Calibration = 0        # if = 1, then 'yes', else 'no'

I = list()
r2 = list()
for i in range(8):
    # % Load the inputs (no user input required)
    if Calibration == 1:
        filename = str("cal_{}".format(i+1))
    else:
        filename = str("val_{}".format(i+1))
    fileid = fileloc + filename + '.txt'
    print("Experiment: " + filename)
    
    # Read the time-series data for the experiment
    obs = pd.read_csv(fileid, header=0, skiprows=3, sep='\t')
    
    # Read the metadata for the experiment [get psi and theta_i]
    with open(fileid) as f:
        meta = np.genfromtxt(itertools.islice(f, 1, 3), delimiter=' = ', usecols=1)
    
    # Input params
    psi = meta[1]
    del_theta = 0.36 - meta[0]
    C = psi * del_theta
    time = obs['time (h)']; T = len(time)
    rRate = obs['Rainfall (mm)'] * 2
    runoffRate = obs['Observed runoff (mm)'] * 2
    infilExpRate = rRate - runoffRate
    infilExpRate[infilExpRate < 0] = 0
    
    Finit = 1.
    F = np.zeros((len(musel), T))
    
    infilRateModel = np.zeros((len(musel), T))
    
    for t in range(T-1):
        F0 = F[:, t]
        # print("Timestep {} of {}".format(t+1, T-1))
        if rRate[t+1] == 0:
            F[:, t+1] = F0
            continue   
        for i in range(len(musel)):
            params = (F0[i], rRate[t+1], time[t], musel[i], sigsel[i], time[t+1], C)
            F[i, t+1] = fsolve(cuminfil, Finit, args=params)
    
    infilRateModel[:, 1:T] = arealinfil(F[:, 1:T], rRate[1:T], musel, sigsel, C)
    Imax = np.max(infilRateModel, axis=0); Imin = np.min(infilRateModel, axis=0)
    I.append([time.values, infilExpRate.values, Imax, Imin, infilRateModel[0,:]])
    r2.append(r2_score(infilExpRate.values[1:-1], infilRateModel[0,1:-1]))


# Create validation plots
import seaborn as sns
from matplotlib.ticker import MultipleLocator      
def val_plot(data, r2, ifsave, inst):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7.3,9), dpi=200)
    for idx in range(len(data)):
        val = data[idx]
        col = 0
        if (idx % 2) == 1:
            col = 1
        ax[idx//2, col].plot(val[0], val[1], 'd', markersize=3, 
          color='r', label='Observed', zorder=3)
        # ax[idx//2, col].plot(val[0], val[4], '-', color='k', lw=1.0, label='Most likely model')
        ax[idx//2, col].fill_between(val[0], val[2], val[3], 
          color=sns.color_palette('PRGn', 10)[8], alpha=0.5, label='95% credible region', lw=0)
        
        ax[idx//2, col].set_xlabel('Time (h)')
        ax[idx//2, col].set_ylabel('$E[I(t)]$ (mm/h)')
        ax[idx//2, col].set_title(r'Event: val_%d' % (idx+1), fontsize=9, weight='bold')
        ax[idx//2, col].set_xlim([-0.1*np.max(val[0])/8.0, np.ceil(np.max(val[0])) + 0.1*np.max(val[0])/8.0])
        if idx == 0:
            ymax = 14
        elif idx == 1:
            ymax = 14
            ax[idx//2, col].legend(loc='upper right', frameon=False)
        else:
            ymax = 10
        ax[idx//2, col].set_ylim([-0.4*ymax/10.0, ymax + 0.4*ymax/10.0])
        # ax[idx//2, col].text(0.05, 0.9, r'$R^2 = {:0.3f}$'.format(r2[idx]), transform=ax[idx//2, col].transAxes, fontdict=font)
        ax[idx//2, col].xaxis.set_major_locator(MultipleLocator(1))
        ax[idx//2, col].xaxis.set_minor_locator(MultipleLocator(0.5))
        if idx == 0:
            ax[idx//2, col].yaxis.set_major_locator(MultipleLocator(4))
        ax[idx//2, col].xaxis.set_tick_params(which='major', length=3, width=0.5, direction='in')
        ax[idx//2, col].xaxis.set_tick_params(which='minor', length=1.5, width=0.5, direction='in')
        ax[idx//2, col].yaxis.set_tick_params(which='major', length=3, width=0.5,direction='in')
    fig.tight_layout()
    if ifsave == 1:
        save_loc = '../figures/coarsening/validation_' + inst + '.png'
        fig.savefig(save_loc, dpi=300)


val_plot(I, r2, ifsave=1, inst=inst)


# Create validation plots
  
# def val_plot_ppt(data, r2, ifsave):
#     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(3.65,4.5), dpi=200)
    
#     for idx in range(len(data)):
#         if idx == 2 or idx == 4:
#             val = data[idx]
#             col = 0
#             ax[idx//4, ].plot(val[0], val[1], 'd', markersize=3, 
#               color='r', label='Observed', zorder=3)
#             ax[idx//4, ].plot(val[0], val[4], '-', color='k', lw=1.0, label='Most likely model')
#             ax[idx//4, ].fill_between(val[0], val[2], val[3], 
#               color=sns.color_palette('PRGn', 10)[8], alpha=0.5, label='95% credible region', lw=0)
#             ax[idx//4, ].legend(loc='upper right', frameon=False)
#             ax[idx//4, ].set_xlabel('Time (h)')
#             ax[idx//4, ].set_ylabel('$E[I(t)]$ (mm/h)')
#             # ax[idx//4, ].set_title(r'Event: val_%d' % (idx+5), fontsize=9, weight='bold')
#             ax[idx//4, ].set_xlim([-0.1*np.max(val[0])/8.0, np.ceil(np.max(val[0])) + 0.1*np.max(val[0])/8.0])
#             if idx == 2:
#                 ymax = 13
#             elif idx == 4:
#                 ymax = 13
#             else:
#                 ymax = 10
#             ax[idx//4, ].set_ylim([-0.4*ymax/10.0, ymax + 0.4*ymax/10.0])
#             # ax[idx//2, col].text(0.05, 0.9, r'$R^2 = {:0.3f}$'.format(r2[idx]), transform=ax[idx//2, col].transAxes, fontdict=font)
#             ax[idx//4, ].xaxis.set_major_locator(MultipleLocator(1))
#             ax[idx//4, ].xaxis.set_minor_locator(MultipleLocator(0.5))
#             if idx == 0:
#                 ax[idx//2, col].yaxis.set_major_locator(MultipleLocator(4))
#             ax[idx//4, ].xaxis.set_tick_params(which='major', length=3, width=0.5, direction='in')
#             ax[idx//4, ].xaxis.set_tick_params(which='minor', length=1.5, width=0.5, direction='in')
#             ax[idx//4, ].yaxis.set_tick_params(which='major', length=3, width=0.5,direction='in')
#         else:
#             continue
#     fig.tight_layout()
#     if ifsave == 1:
#         fig.savefig(r'D:\Research\2021\CONFERENCES\EWRI 2021\Figures\validation_with_gmm85.png', dpi=600)


# val_plot_ppt(I, r2, ifsave=0)
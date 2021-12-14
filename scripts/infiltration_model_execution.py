# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:02 2020

@author: abhishek


This script executes the infiltration model from Govindaraju et al (2001) and Goyal et al (2019).

Companion scripts:
    1. infiltration_model.py
    2. colorplots.py

"""

# %% Load the required modules
import numpy as np
from scipy.special import erfc
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from numpy import sqrt
import itertools
from scipy.interpolate import interp1d, interp2d
import scipy.signal

from infiltration_model import *
from colorplots import colorplot, colorplot_single

# %% User inputs
exp = input("Experiment Number: ")

fileloc = r'../dataset/experiment_data/'
exp_no = int(exp)             # Experiment number
Calibration = 1        # if = 1, then 'yes', else 'no'
n_muy = 100           # number of muy samples 
n_sigy = 100            # number of sigy samples 

# %% Load the inputs (no user input required)
if Calibration == 1:
    filename = str("cal_{}".format(exp_no))
else:
    filename = str("val_{}".format(exp_no))
fileid = fileloc + filename + '.txt'

# Read the time-series data for the experiment
obs = pd.read_csv(fileid, header=0, skiprows=3, sep='\t')

# Read the metadata for the experiment [get psi and theta_i]
with open(fileid) as f:
    meta = np.genfromtxt(itertools.islice(f, 1, 3), delimiter=' = ', usecols=1)

# import point measurement data
fin = pd.read_csv(r'../dataset/experiment_data/kspoint.csv', 
                    skiprows=1, sep=',', header=0, engine='python')
pointData = fin.gp
D1 = fin.dri
# pointData = pointData.append(fin.gp, ignore_index=True)
# pointData = pointData.append(fin.csiro, ignore_index=True)


# D1t = 19.92*np.log(D1/47.422)
D1t = 0.0236*D1 + 0.9338
pointData = pointData.append(D1t, ignore_index=True)
# pointData = -76.874 + 19.92*np.log(pointData)
# pointData = pointData.append(fin.gp, ignore_index=True)


pointData = pointData.values
pointData = np.delete(pointData, np.where(pointData < 0))


# Input params
psi = meta[1]
del_theta = 0.36 - meta[0]
C = psi * del_theta
time = obs['time (h)']; T = len(time)
rRate = obs['Rainfall (mm)'] * 2
runoffRate = obs['Observed runoff (mm)'] * 2
infilExpRate = rRate - runoffRate
infilExpRate[infilExpRate < 0] = 0

# The range of mu_y and sigma_y parameters
mu = np.linspace(-5, 5, n_muy)
sig = np.linspace(0.01, 5, n_sigy)
dx2 = (mu[1]-mu[0])*(sig[1]-sig[0])
X, Y = np.meshgrid(mu, sig)
X1 = np.ravel(X)
Y1 = np.ravel(Y)

# %% Define the prior distribution

def prior(point_data, x, y):
    n = 1000
    t = np.linspace(-10, 1000, n)
    
    # Measurement error distribution (assuming a fixed std. dev.)
    errscale = 5.0
    errpdf = stats.norm.pdf(x=t, loc=0, scale=errscale)
    comb_lhood = np.zeros((len(x),))
    # ppdf = np.zeros((1999, len(x)))
    for idx in range(len(x)):
        KPointpdf = stats.lognorm.pdf(x=t, s=y[idx], scale=np.exp(x[idx]))
        conv_pdf = scipy.signal.fftconvolve(errpdf, KPointpdf, mode='full')
        t_full = np.linspace(t[0] + t[0], t[-1] + t[-1], len(conv_pdf))
        if np.any(conv_pdf):
            conv_pdf /= np.trapz(conv_pdf, x=t_full)
    #     ppdf[:, idx] = conv_pdf
    # return ppdf, t_full
        
        f = interp1d(t_full, conv_pdf)
        lhood = f(point_data)
        lhood[np.isnan(lhood)] = 1e-20
        lhood[lhood <= 0] = 1e-20
        comb_lhood[idx,] = np.sum(np.log(lhood))
    return np.exp(comb_lhood)

ppdf = prior(pointData, X1, Y1)
ppdf = ppdf/(np.sum(ppdf)*dx2)

# fig, ax = plt.subplots(figsize=(6,4), dpi=300)
# for i in range(len(t_full)):
#     ax.plot(t_full, ppdf[:,i])
colorplot_single(X, Y, ppdf.reshape(X.shape), ifsave=0)

# p = interp2d(X1, Y1, ppdf, kind='linear', bounds_error=False, fill_value=None)
# pnew = p(mu, sig)
# colorplot_single(X, Y, p(mu, sig), ifsave=0)

# %% Execution
Finit = 1.
F = np.zeros((len(X1), T))

infilRateModel = np.zeros((len(X1), T))
pdf = np.ones((len(X1), T))
for t in range(T-1):
    F0 = F[:, t]
    print("Timestep {} of {}".format(t+1, T-1))
    if rRate[t+1] == 0:
        F[:, t+1] = F0
        continue   
    for i in range(len(X1)):
        params = (F0[i], rRate[t+1], time[t], X1[i], Y1[i], time[t+1], C)
        F[i, t+1] = fsolve(cuminfil, Finit, args=params)

infilRateModel[:, 1:T] = arealinfil(F[:, 1:T], rRate[1:T], X1, Y1, C)
pdf[:, 1:T] = stats.norm.pdf(x=infilRateModel[:, 1:T], loc=infilExpRate[1:T], scale=1.)
# pdf[:, 0] = prior(pointData, X1, Y1)
pdf[:, 0] = ppdf

# Calculate the likelihood of each parameter combination
# Parameter combinations are rows and the time-steps are columns
likelihood = np.cumprod(pdf, axis=1)        # this computes the likelihood at each time-step
npdf = np.divide(likelihood, (np.sum(likelihood, axis=0)*dx2))  # normalize the likelihood to get a pdf

# Reshape the npdf matrix to create the contour maps
npdf1 = np.zeros((len(mu), len(sig), T))
npdf1 = np.zeros((len(sig), len(mu), T))
for idx in range(T):
    npdf1[:,:, idx] = npdf[:, idx].reshape(X.shape)
# np.save(r'../variables/cal2_{}'.format(exp_no), npdf1)       # save the outcome

# colorplot(time, X, Y, npdf1, exp_no, ifsave=0)              # create a colorplot
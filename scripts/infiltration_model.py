# -*- coding: utf-8 -*-
"""
Infiltration module

Created on Mon Mar  9 14:49:00 2020

@author: abhishek
"""

import numpy as np
from numpy import sqrt
from scipy.special import erfc

def G(var, csi, muy, sigy):
    return np.exp(csi * muy + np.power((sigy * csi),2)/2) * (1 - 0.5 * erfc(var - sigy * csi/2**(0.5)))

# Simulation of cumulative infiltration, F
def cuminfil(F, *params):
    F0, r, t0, muy, sigy, t1, C = params
    eps = 1e-10   
    if F > 0:
        F1 = F
    else:
        F1 = eps   
    temp = (np.log(F1*r/(C+F1)) - muy)/(sqrt(2)*sigy)      
    s = 0
    for i in range(1, 6):
        s = s + G(temp, i, muy, sigy)/((i+1)*r**(i+1)) 
    return t0 + (F1/r)*(1 - G(temp, 0, muy, sigy)) - (F0/r) + (F1 + C*np.log(C/(C+F1)))*G(temp, -1, muy, sigy) + C * s - t1

# Simulation of areal infiltration rate and likelihood

def arealinfil(F, r, muy, sigy, C):
    r = r[:, np.newaxis]
    muy = muy[:, np.newaxis]
    sigy = sigy[:, np.newaxis]
    temp1 = np.divide(F, (C+F))
    Kc = temp1 * r.T
    Kc[Kc == 0] = 1e-10
    temp2 = np.divide((np.log(Kc) - muy), (sqrt(2) * sigy))
    return (1-G(temp2, 0, muy, sigy))*r.T + np.divide(G(temp2, 1, muy, sigy), temp1)
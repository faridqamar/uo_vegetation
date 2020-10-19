#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pysmarts2
from scipy.interpolate import interp1d


def modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH,
              W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, 
              ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, 
              AbO2, AbBrO, AbClNO, TAU5):
# -- Function to call pySMARTS and produce a model
    nalb = 111
    mywav = np.linspace(0.35,0.9,nalb)
    np.around(mywav, 2, mywav)
    albedo = albedoFunc(mywav, a1, b1, c1, a2, b2, c2, a3, b3, c3, d)
    err_set = np.seterr(all='ignore')
    np.around(albedo, 4, albedo)

    
    if scan == '108':
        Year = 2016
        Month = 5
        Day = 5
        Hour = 14.02
        TAIR = 15.5
        #RH = 69.0
        TDAY = 12.5
    elif scan == '000':
        Year = 2016
        Month = 5
        Day = 2
        Hour = 17.77
    
    albwav = np.zeros(shape=(3000))
    albalb = np.zeros(shape=(3000))
    l = np.zeros(shape=(14,636))
    albwav[:nalb] = mywav
    albalb[:nalb] = albedo
    
    pymod = pysmarts2.smarts295(TAIR, RH, TDAY, W, ApCH2O, ApCH4, ApCO, ApHNO2, 
                               ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, 
                               qCO2, AbO2, AbBrO, AbClNO, TAU5,
                               1, 1, albwav, albalb, nalb, Year, Month, Day, Hour, l)
    
    return pymod[0], pymod[-2]


def albedoFunc(wav, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
# -- Function to produce an albedo array
    err_set = np.seterr(all='raise')
    try:
        albedo = (b1*np.exp(-((wav-a1)**2)/(2*(c1**2)))) + (b2*np.exp(-((wav-a2)**2)/(2*(c2**2)))) + \
                (b3*np.exp(-((wav-a3)**2)/(2*(c3**2)))) + d
    except:
        albedo = np.full(len(wav), -np.inf)
        
    return np.array(albedo)



def interpModel(mywav, amp, modelwav, modelsmrt):
# -- Function to interpolate the pySMARTS model into the cube's wavelengths
#    and multiply by the given amplitude
    err_set = np.seterr(all='raise')
    try:
        interpMod = interp1d(modelwav, modelsmrt, fill_value="extrapolate")
        model = np.array(interpMod(mywav)) * amp
    except:
        model = np.full(len(mywav), -np.inf)
    
    return model



# -- Defining MCMC functions
def log_prior(theta, wav, scan):
#    a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d, \
#    W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, \
#    ApSO2, qCO2, TAU5, amp, eps = theta
    
#    a1, b1, c1, a2, b2, c2, a3, b3, c3, d, eps = theta
#    W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5, amp, eps = theta
#    W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5, amp, eps = theta
    a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH, W, ApHNO2, ApNO2, ApNO3, \
    AbO3, ApO3, ApSO2, AbO2, TAU5, eps = theta
    
    if eps <= 0:
        return -np.inf
    if (a1 < 0.53) or (a1 >= 0.65):
        return -np.inf
    if (a2 < 0.65 ) or (a2 >= 0.73):
        return -np.inf
#    if (a3 < 0.8 ) or (a3 >= 200.0):
#        return -np.inf
    if (a3 < 0.4) or (a3 >= 0.53):
        return -np.inf
    if (b1 < 0.0) or (b1 > 0.7):
        return -np.inf
    if (b2 < 0.12) or (b2 > 0.25):
        return -np.inf
#    if (b3 < 0.0) or (b3 > 30.0):
#        return -np.inf
    if (b3 <= 0.0) or (b3 > 0.3):
        return -np.inf
    if (c1 <= 0.0) or (c1 > 0.1): 
        return -np.inf
    if (c2 <= 0.04) or (c2 > 0.08): 
        return -np.inf
#    if (c3 <= 0.0) or (c3 > 200.0): 
#        return -np.inf
    if (c3 <= 0.0) or (c3 > 0.1): 
        return -np.inf
#    if (amp <= 0):
#        return -np.inf
    nwav = np.linspace(0.35,0.9,111)
    albedo = albedoFunc(nwav, a1, b1, c1, a2, b2, c2, a3, b3, c3, d)
    if any(np.isnan(albedo)) or not any(np.isfinite(albedo)):
        return -np.inf
    if (any(albedo) < 0) or (any(albedo) > 1):
        return -np.inf
#    if (TAIR < -10) or (TAIR > 40):
#        return -np.inf
    if (RH < 0) or (RH > 100):
        return -np.inf
#    if (TDAY < -10) or (TDAY > 40):
#        return -np.inf
    if (W < 0) or (W > 12):
        return -np.inf
#    if (ApCH2O < 0) or (ApCH2O > 5.0):
#        return -np.inf
#    if (ApCH4 < 0) or (ApCH4 > 5.0):
#        return -np.inf
#    if (ApCO < 0) or (ApCO > 5.0):
#        return -np.inf
    if (ApHNO2 < 0) or (ApHNO2 > 5.0):
        return -np.inf
#    if (ApHNO3 < 0) or (ApHNO3 > 5.0):
#        return -np.inf
#    if (ApNO < 0) or (ApNO > 5.0):
#        return -np.inf
    if (ApNO2 < 0) or (ApNO2 > 5.0):
        return -np.inf
    if (ApNO3 < 0) or (ApNO3 > 5.0):
        return -np.inf
    if (AbO3 < 0) or (AbO3 > 5.0):
        return -np.inf
    if (ApO3 < 0) or (ApO3 > 5.0):
        return -np.inf
    if (ApSO2 < 0) or (ApSO2 > 5.0):
        return -np.inf
#    if (qCO2 < 0) or (qCO2 > 1000):
#        return -np.inf
    if (AbO2 < 0) or (AbO2 > 1e7):
        return -np.inf
#    if (AbO4 < 0) or (AbO4 > 1e45):
#        return -np.inf
    if (TAU5 < 0) or (TAU5 > 5.57):
        return -np.inf

#    a1 = 0.62
#    b1 = 0.159
#    c1 = 0.114

#    a2 = 0.755
#    b2 = 0.0748
#    c2 = 0.045

#    a3 = 1.9
#    b3 = 0.111
#    c3 = 1.049

#    a4 = 0.584
#    b4 = 0.07
#    c4 = 0.11
#    d  = 0.09

    ApCH2O = 0.0
    ApCH4  = 0.0
    ApCO   = 0.0
    ApHNO3 = 0.0
    ApNO   = 0.0
    qCO2   = 0.0
    AbBrO  = 2.5e-6
    AbClNO = 0.00012
    amp    = 2000.0
    
#    W = 2.0
##    ApCH2O = 0.007
##    ApCH4 = 0.3
##    ApCO = 0.35
#    ApHNO2 = 0.002
##    ApHNO3 = 0.005
##    ApNO = 0.2
#    ApNO2 = 0.02
#    ApNO3 = 5e-5
#    ApO3 = 0.053
#    AbO3 = 0.33
#    ApSO2 = 0.05
##    qCO2 = 370.0
#   AbBrO  = 2.5e-6
#    AbClNO = 0.00012
#    TAU5 = 0.084

    modwav, modsmrt = modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH,
                                W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3,
                                ApSO2, qCO2, AbO2, AbBrO, AbClNO, TAU5)
    if any(np.isnan(modsmrt)) or not any(np.isfinite(modsmrt)):
        modwav, modsmrt = modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH,
                                    W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, 
                                    ApSO2, qCO2, AbO2, AbBrO, AbClNO, TAU5)
        if any(np.isnan(modsmrt)) or not any(np.isfinite(modsmrt)):
            return -np.inf
    
    model = interpModel(wav, amp, modwav, modsmrt)
    if (any(model) < 0) or any(np.isnan(model)) or not any(np.isfinite(model)):
        return -np.inf
    return 0.0



def log_likelihood(theta, wav, y, scan):  
#    a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d, \
#    W, ApCH2O, ApCH4, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, \
#    ApSO2, qCO2, TAU5, amp, eps = theta
    
#    a1, b1, c1, a2, b2, c2, a3, b3, c3, d, eps = theta
#    W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5, amp, eps = theta
#    W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5, amp, eps = theta
#    a1, b1, c1, a2, b2, c2, a3, b3, c3, d, W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5, eps = theta
    a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH, W, ApHNO2, ApNO2, ApNO3, \
    AbO3, ApO3, ApSO2, AbO2, TAU5, eps = theta

#    a1 = 0.62
#    b1 = 0.159
#    c1 = 0.114

#    a2 = 0.755
#    b2 = 0.0748
#    c2 = 0.045

#    a3 = 1.9
#    b3 = 0.111
#    c3 = 1.049

#    a4 = 0.584
#    b4 = 0.07
#    c4 = 0.11
#    d  = 0.09

    ApCH2O = 0.0
    ApCH4  = 0.0
    ApCO   = 0.0
    ApHNO3 = 0.0
    ApNO   = 0.0
    qCO2   = 0.0
    AbBrO  = 2.5e-6
    AbClNO = 0.00012
    amp = 2000.0
    
#    W = 2.0
##    ApCH2O = 0.007
##    ApCH4 = 0.3
##    ApCO = 0.35
#    ApHNO2 = 0.002
##    ApHNO3 = 0.005
##    ApNO = 0.2
#    ApNO2 = 0.02
#    ApNO3 = 5e-5
#    ApO3 = 0.053
#    AbO3 = 0.33
#    ApSO2 = 0.05
##    qCO2 = 370.0
#   AbBrO  = 2.5e-6
#    AbClNO = 0.00012
#    TAU5 = 0.084
    
    modwav, modsmrt = modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH,
                                W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3,
                                ApSO2, qCO2, AbO2, AbBrO, AbClNO, TAU5)
    if any(np.isnan(modsmrt)) or not any(np.isfinite(modsmrt)):
        modwav, modsmrt = modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d, RH,
                                    W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, 
                                    ApSO2, qCO2, AbO2, AbBrO, AbClNO, TAU5)
        if any(np.isnan(modsmrt)) or not any(np.isfinite(modsmrt)):
            return -np.inf
    
    model = interpModel(wav, amp, modwav, modsmrt)
    if any(np.isnan(model)) or not any(np.isfinite(model)) or (any(model) < 0):
        return -np.inf
        
    denom = eps**2
    lk = -0.5 * sum(((y-model)**2) / denom + np.log(denom) + np.log(2*np.pi))
    return lk

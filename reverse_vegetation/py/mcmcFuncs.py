#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pysmarts
from scipy.interpolate import interp1d


def modelFunc(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4,
              TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3, ApNO2, ApNO3, 
              ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5):
# -- Function to call pySMARTS and produce a model
    nalb = 101
    mywav = np.linspace(0.3,1.3,nalb)
    np.around(mywav, 2, mywav)
    albedo = albedoFunc(mywav, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4)
    np.around(albedo, 4, albedo)

    #TAIR = 15.5
    #RH = 69.0
    #TDAY = 12.5
    Year = 2016
    Month = 5
    Day = 5
    Hour = 14.02
    
    albwav = np.zeros(shape=(3000))
    albalb = np.zeros(shape=(3000))
    l = np.zeros(shape=(14,636))
    albwav[:nalb] = mywav
    albalb[:nalb] = albedo
    
    pymod = pysmarts.smarts295(TAIR,RH,TDAY, W, AbO3, ApCH2O, 0.0, 0.0, ApHNO2, 
                             ApHNO3, 0.0, ApNO2, ApNO3, ApO3, ApSO2, 0.0, 
                             ALPHA1, ALPHA2, OMEGL, GG, TAU5, 1, 1,
                             albwav, albalb, nalb, Year, Month, Day, Hour, l)
    
    return pymod[0], pymod[-2]


def albedoFunc(wav, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4):
# -- Function to produce an albedo array
    albedo = ((b1/(2*np.pi))*np.exp(-((wav-a1)**2)/(2*c1))+d1) + ((b2/(2*np.pi))*np.exp(-((wav-a2)**2)/(2*c2))+d2) + \
    ((b3/(2*np.pi))*np.exp(-((wav-a3)**2)/(2*c3))+d3) + ((b4/(2*np.pi))*np.exp(-((wav-a4)**2)/(2*c4))+d4)
    
    return np.array(albedo)


def interpModel(mywav, amp, modelwav, modelsmrt):
# -- Function to interpolate the pySMARTS model into the cube's wavelengths
#    and multiply by the given amplitude
    interpMod = interp1d(modelwav, modelsmrt, fill_value="extrapolate")
    model = np.array(interpMod(mywav)) * amp
    
    return model

# -- Defining MCMC functions
def log_prior(theta, wav, model):
    a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4, \
    TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3, ApNO2, ApNO3,  \
    ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5, amp, eps = theta
    if eps <= 0:
        print("**EPS = ", eps)
        return -np.inf
    if (c1 == 0) or (c2 == 0) or (c3 == 0) or (c4 == 0):
        print("**C = ", c1, c2, c3, c4)
        return -np.inf
    if (amp <= 0):
        print("**Amplitude = ", amp)
        return -np.inf
    nwav = wav/1000.
    albedo = albedoFunc(nwav, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4)
    if (any(albedo) < 0) or (any(albedo) > 1):
        print("**ALBEDO not in 0-1 range:", a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4)
        return -np.inf
    if (TAIR > 40) or (TAIR < -25):
        print("**TAIR = ", TAIR)
        return -np.inf
    if (RH < 0) or (RH > 100):
        print("**RH = ", RH)
        return -np.inf
    if (TDAY > 40) or (TDAY < -25):
        print("**TDAY = ", TDAY)
        return -np.inf
    if (W < 0) or (W > 12):
        print("**W = ", W)
        return -np.inf
    if (AbO3 < 0) or (AbO3 > 5.0):
        print("**AbO3 = ", AbO3)
        return -np.inf
    if (ApCH2O < 0) or (ApCH2O > 5.0):
        print("**ApCH2O = ", ApCH2O)
        return -np.inf
    if (ApHNO2 < 0) or (ApHNO2 > 5.0):
        print("**ApHNO2 = ", ApHNO2)
        return -np.inf
    if (ApHNO3 < 0) or (ApHNO3 > 5.0):
        print("**ApHNO3 = ", ApHNO3)
        return -np.inf
    if (ApNO2 < 0) or (ApNO2 > 5.0):
        print("**ApNO2 = ", ApNO2)
        return -np.inf
    if (ApNO3 < 0) or (ApNO3 > 5.0):
        print("**ApNO3 = ", ApNO3)
        return -np.inf
    if (ApO3 < 0) or (ApO3 > 5.0):
        print("**ApO3 = ", ApO3)
        return -np.inf
    if (ApSO2 < 0) or (ApSO2 > 5.0):
        print("**ApSO2 = ", ApSO2)
        return -np.inf
    if (ALPHA1 < 0) or (ALPHA1 > 5.0):
        print("**ALPHA1 = ", ALPHA1)
        return -np.inf
    if (ALPHA2 < 0) or (ALPHA2 > 5.0):
        print("**ALPHA2 = ", ALPHA2)
        return -np.inf
    if (OMEGL < 0.001) or (OMEGL > 1.0):
        print("**OMEGL = ", OMEGL)
        return -np.inf
    if (GG < 0.001) or (GG > 1.0):
        print("**GG = ", GG)
        return -np.inf
    if (TAU5 < 0) or (TAU5 > 1.0):
        print("**TAU5 = ", TAU5)
        return -np.inf
    if (any(model) < 0):
        print("**Model has < 0")
        return -np.inf
    return 0.0

def log_likelihood(eps, model, y):  
    denom = eps**2
    lp = -0.5 * sum(((y-model)**2) / denom + np.log(denom) + np.log(2*np.pi))
    return lp

def log_probability(theta, wav, y):
    a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4, \
    TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3, ApNO2, ApNO3,  \
    ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5, amp, eps = theta
    
    modwav, modsmrt = modelFunc(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4,
                                TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3, ApNO2, ApNO3,
                                ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5)
    if any(np.isnan(modsmrt)):
        modwav, modsmrt = modelFunc(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4,
                                    TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3, ApNO2, ApNO3,
                                    ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5)
        if any(np.isnan(modsmrt)):
            print("**MODEL has NaN:", theta)
            print("LOG_PROBABILITY = ", -np.inf)
            return -np.inf
        
    model = interpModel(wav, amp, modwav, modsmrt)
    
    lp = log_prior(theta, wav, model)
    if not np.isfinite(lp):
        print("LOG_PROBABILITY = ", -np.inf)
        return -np.inf
    
    lk = log_likelihood(eps, model, y)
    lprb = lp + lk
    print("LOG_PROBABILITY = ", lprb)
    return lprb

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import hyss_util as hu
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import mcmcFuncs as mc
import pysmarts
import emcee
import corner
from multiprocessing import Pool



#def modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
#def modelFunc(scan, W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, 
#              ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5):
#def modelFunc(scan, W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5):
def modelFunc(scan, a1, b1, c1, a2, b2, c2, a3, b3, c3, 
              d, W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5):
# -- Function to call pySMARTS and produce a model

#    a1 = 0.62
#    b1 = 0.159
#    c1 = 0.114
#    d1 = 0.10

#    a2 = 0.755
#    b2 = 0.0748
#    c2 = 0.045
#    d2 = -0.01

#    a3 = 1.9
#    b3 = 0.111
#    c3 = 1.049
#    d3 = 0.0001

#    a4 = 0.584
#    b4 = 0.07
#    c4 = 0.11
#    d4 = 0.0001

    nalb = 111
    mywav = np.linspace(0.35,0.9,nalb)
    np.around(mywav, 2, mywav)
    albedo = mc.albedoFunc(mywav, a1, b1, c1, a2, b2, c2, a3, b3, c3, d)
    err_set = np.seterr(all='ignore')
    np.around(albedo, 4, albedo)

#    W = 2.0
##    ApCH2O = 0.007
    ApCH2O = 0.0
    ApCH4 = 0.0
    ApCO = 0.0
#    ApHNO2 = 0.002
##    ApHNO3 = 0.005
    ApHNO3 = 0.0
    ApNO = 0.0
#    ApNO2 = 0.02
#    ApNO3 = 5e-5
#    AbO3 = 0.33
#    ApO3 = 0.053
#    ApSO2 = 0.05
    qCO2 = 0.0
#    TAU5 = 0.084
    
    if scan == '108':
        Year = 2016
        Month = 5
        Day = 5
        Hour = 14.02
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
    
    pymod = pysmarts.smarts295(W, ApCH2O, ApCH4, ApCO, ApHNO2, 
                               ApHNO3, ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5, 
                               1, 1, albwav, albalb, nalb, Year, Month, Day, Hour, l)
    
    return pymod[0], pymod[-2]



# --  set scan number:
scan = '108'

# -- read the cube from .raw file into float array
fname = "../../../image_files/veg_00" + scan +".raw"
cube = hu.read_hyper(fname)
#cube_sub = cube.data[:, :, :].astype(float)

# -- choosing 5x5 pixels in building adjacent to vegetation
#    taking mean of spectra to use for fitting

print("Picking building pixels ...")
brow = np.arange(1015,1020)
bcol = np.arange(847,852)
bcoords = np.vstack(np.meshgrid(brow, bcol)).reshape(2,-1).T
blds = cube.data[:,1015:1020,847:852].mean(axis=(1,2))
#for coord in bcoords:
#    plt.plot(cube.waves, cube.data[:,coord[0],coord[1]], lw=0.5)
#plt.plot(cube.waves, blds, color='black')
#plt.show()


# -- multiplying mean building spectrum by wavelength
print("Multiplying mean building spectrum by wavelength ...")
nblds = blds*cube.waves/1e3
#plt.plot(cube.waves, blds, label='raw spectrum')
#plt.plot(cube.waves, nblds, label='spectrum*wavelength')
#plt.legend()
#plt.show()

mywav = cube.waves[:-200]
myblds = nblds[:-200]

nwalkers, ndim = 200, 19

flat_samples = np.load("../output/flat_samples_108.npy")

# -- Corner Plots
labels = [r'$\mu_1$', r'$b_1$', r'$\sigma_1$',
          r'$\mu_2$', r'$b_2$', r'$\sigma_2$',
          r'$\mu_3$', r'$b_3$', r'$\sigma_3$', 'd',
          r'$H_2O$', r'$HNO_2$', r'$NO_2$', r'$NO_3$', r'$AbO_3$', r'$ApO_3$', r'$SO_2$', r'$\tau_5$', 'eps']

#print("")
#print("Plotting Albedo Corner Plot ...")
#f, ax = plt.subplots(10, 10, figsize=((10)*2,(10)*2))
#fig = corner.corner(flat_samples[:,:10], labels=labels[:10], truths=np.median(flat_samples[:,:10], axis=0), fig=f)
#f.canvas.draw()
#f.savefig("../output/MCMC_Corner_Albedo_"+scan+".png", dpi=300)

#print("")
#print("Plotting Atmosphere Corner Plot ...")
#f, ax = plt.subplots(8, 8, figsize=((8)*2,(8)*2))
#fig = corner.corner(flat_samples[:,10:-1], labels=labels[10:-1], truths=np.median(flat_samples[:,10:-1], axis=0), fig=f)
#f.canvas.draw()
#f.savefig("../output/MCMC_Corner_Atmos_"+scan+".png", dpi=300)


plt.style.use('seaborn-ticks')
plt.rcParams["axes.facecolor"]  = "w"
plt.rcParams["axes.grid"]       = True
plt.rcParams["axes.labelcolor"] = "k"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"]      = "#444444"
plt.rcParams["grid.linewidth"]  = 1.5
plt.rcParams["grid.linestyle"]  = ":"
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.color"]     = "k"
plt.rcParams["ytick.color"]     = "k"
plt.rcParams["text.color"]      = "k"
plt.rcParams["font.size"]       = 20
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

amp = 2000.0
# -- Plot a sample of the MCMC solutions
print("")
print("Plotting MCMC Sample Solutions ...")
fig, ax = plt.subplots(figsize=(10,6))
inds = np.random.randint(len(flat_samples), size=5000)
for ind in inds:
    sample = flat_samples[ind]
    #smrtwav, smrtmod = modelFunc(scan, *sample[:-2])
    #maxmod = mc.interpModel(mywav, sample[-2], smrtwav, smrtmod)
    smrtwav, smrtmod = modelFunc(scan, *sample[:-1])
    maxmod = mc.interpModel(mywav, amp, smrtwav, smrtmod)
    linm, = ax.plot(mywav, maxmod, color='dodgerblue', lw=0.3)
linb, = ax.plot(mywav, myblds, color='darkred')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Intensity [arbitrary units]')
ax.legend([linb, linm], ['data', 'model'])
fig.savefig("../output/MCMC_models_"+scan+".png", dpi=300)


print("")
print("Plotting MCMC Albedo Solutions ...")
fig, ax = plt.subplots(figsize=(10,6))
inds = np.random.randint(len(flat_samples), size=5000)
for ind in inds:
    sample = flat_samples[ind]
    albedo = mc.albedoFunc(mywav/1000., *sample[:10])
    linm, = ax.plot(mywav, albedo, color='dodgerblue', lw=0.2)
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Albedo')
#    ax.set_ylim(-0.1,0.6)
fig.savefig("../output/MCMC_albedo_"+scan+".png", dpi=300)


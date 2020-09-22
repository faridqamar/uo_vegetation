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

# -- Setting initial parameter values
print("Initializing parameters ...")
a1 = 0.62
b1 = 1.0
c1 = 0.013
d1 = 0.10

a2 = 0.755
b2 = 0.47
c2 = 0.002
d2 = -0.01

a3 = 1.9
b3 = 0.7
c3 = 1.1
d3 = 0.0001

a4 = 0.584
b4 = 0.35
c4 = 0.01
d4 = 0.0001

TAIR = 15.5
RH = 69.0
TDAY = 12.5

W = 2.0
AbO3 = 0.33
ApCH2O = 0.007
#ApCH4 = 0.3
#ApCO = 0.35
ApHNO2 = 0.002
ApHNO3 = 0.005
#ApNO = 0.2
ApNO2 = 0.02
ApNO3 = 5e-5
ApO3 = 0.053
ApSO2 = 0.05
#qCO2 = 370.0
ALPHA1 = 0.9111
ALPHA2 = 1.3529
OMEGL = 0.8
GG = 0.7
TAU5 = 0.084
    
amp = 1999.5
eps = 21.8

init_params = np.array([a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3,
                    a4, b4, c4, d4, TAIR, RH, TDAY, W, AbO3, ApCH2O, ApHNO2, ApHNO3,
                    ApNO2, ApNO3, ApO3, ApSO2, ALPHA1, ALPHA2, OMEGL, GG, TAU5, amp, eps])
print("   initial parameters = ", init_params)


# -- Setting walkers, number of steps, and initial array
nwalkers, ndim, nsteps = 100, init_params.shape[0], 500
p0 = init_params * (1 + np.random.randn(nwalkers, ndim)/1000.)


# -- Setting up backend to save and monitor progress
filename = "MCMC_"+scan+".h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


# -- cut last 200 entries from spectrum
mywav = cube.waves[:-200]
myblds = nblds[:-200]

# -- Perform MCMC
print("Starting MCMC:")
print("   number of walkers    = ", nwalkers)
print("   number of dimensions = ", ndim)
print("   number of steps      = ", nsteps)
start_time = time.time()

#with Pool() as pool:
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)#, pool=pool)
sampler = emcee.EnsembleSampler(nwalkers, ndim, mc.log_probability, args=(mywav, myblds), backend=backend)
sampler.run_mcmc(p0, nsteps)

elapsed_time = time.time() - start_time
print("Total MCMC time = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print("   number of walkers    = ", nwalkers)
print("   number of dimensions = ", ndim)
print("   number of steps      = ", nsteps)


# -- Plot walkers
fig, axes = plt.subplots(ndim, sharex=True, figsize=(8,40))
labels = ['a1', 'b1', 'c1', 'd1',
          'a2', 'b3', 'c2', 'd2',
          'a3', 'b3', 'c3', 'd3',
          'a4', 'b4', 'c4', 'd4',
          'T_air', 'R.Humid', 'T_day',          
          'H2O', 'AbO3', 'ApCH2O', 'ApHNO2', 'ApHNO3', 'ApNO2', 'ApNO3',
          'ApO3', 'ApSO2', 'ALPHA1', 'ALPHA2', 'OMEGL', 'GG', 'TAU5', 'amp', 'eps']
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i], "k", alpha=0.3)
    ax.set_xlim(0,len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1,0.5)

axes[-1].set_xlabel("step number")
fig.canvas.draw()
fig.savefig("../output/MCMC_walkers_"+scan+".png", dpi=300)


# -- Autocorrelation time, burn-in, and flattening
tau = sampler.get_autocorr_time()
burnin = int(3 * np.max(tau))
thin = int(0.5 * np.min(tau))
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
log_prior_samples = sampler.get_blobs(discard=burnin, thin=thin, flat=True)

print("Autocorrelation Time = ", tau)
print("burn-in = ", burnin)
print("thin = ", thin)
print("flat chain shape: ", flat_samples.shape)
print("flat log prob shape: ", log_prob_samples.shape)
print("flat log prior shape: ", log_prior_samples.shape)

all_samples = np.concatenate(
    (flat_samples, log_prob_samples[:,None], log_prior_samples[:,None]), axis=1)


# -- Corner Plot
f, ax = plt.subplots(37, 37, figsize=(74,74))
labels = ['a1', 'b1', 'c1', 'd1',
          'a2', 'b3', 'c2', 'd2',
          'a3', 'b3', 'c3', 'd3',
          'a4', 'b4', 'c4', 'd4',
          'H2O', 'AbO3', 'ApCH2O', 'ApCO', 'ApHNO2', 'ApHNO3', 'ApNO', 'ApNO2', 'ApNO3',
          'ApO3', 'ApSO2', 'qCO2', 'ALPHA1', 'ALPHA2', 'OMEGL', 'GG', 'TAU5', 'amp', 'eps',
          'log prob', 'log prior']
fig = corner.corner(all_samples, labels=labels, truths=np.median(all_samples, axis=0), fig=f)
f.canvas.draw()
f.savefig("../output/MCMC_Corner_"+scan+".png", dpi=300)


# -- Plot a sample of the MCMC solutions
fig = plt.subplots(figsize=(10,6))
inds = np.random.randint(len(flat_samples), size=800)
for ind in inds:
    sample = flat_samples[ind]
    smrtwav, smrtmod = mc.modelFunc(*sample[:-2])
    maxmod = mc.interpModel(cube.waves, sample[-2], modwav, modsmrt)
    linm, = plt.plot(cube.waves, maxmod, color='dodgerblue', lw=0.3)
linb, = plt.plot(cube.waves, nblds, color='darkred')
plt.xlabel('wavelength [nm]')
plt.legend([linb, linm], ['data', 'model'])
fig.savefig("../output/MCMC_models_"+scan+".png", dpi=300)




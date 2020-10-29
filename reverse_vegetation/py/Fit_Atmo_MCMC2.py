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
from mcmcFuncs2 import *
import pysmarts2
import emcee
import corner
from multiprocessing import Pool

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
# -- cut last 255 entries from spectrum
mywav = cube.waves[:-260]
bld = blds[:-260]

# -- Reading quantum efficiency and fixing building spectrum
print("Reading quantum efficiency ...")
qeff = pd.read_csv('./hsi1_qe.csv')
int_qeff= interp1d(qeff['wavelength_nm'], qeff['quantum_efficiency'], fill_value="extrapolate")
iqeff = np.array(int_qeff(mywav))
print("Dividing spectrum by QE and multiplying by wavelength^2 ...")
myblds = bld * mywav * mywav * 2e-10 / (iqeff/100)

# -- multiplying mean building spectrum by wavelength
#print("Multiplying mean building spectrum by wavelength ...")
#nblds = blds*cube.waves/1e3
#plt.plot(cube.waves, blds, label='raw spectrum')
#plt.plot(cube.waves, nblds, label='spectrum*wavelength')
#plt.legend()
#plt.show()

# -- Setting initial parameter values
print("Initializing parameters ...")
#a1 = 0.2
#b1 = 0.5
#c1 = 0.7

a2 = 0.58
b2 = 0.03
c2 = 0.03

a3 = 0.75
b3 = 0.1
c3 = 0.06

#a1 = 0.45
#b1 = 0.2
#c1 = 0.15

#a2 = 0.55
#b2 = 0.05
#c2 = 0.07

#a3 = 0.65
#b3 = 0.03
#c3 = 0.04

#a4 = 0.78
#b4 = 0.43
#c4 = 0.1

d  = 0.25

TAIR = 15.5
RH = 30.0
TDAY = 12.5

W = 1.35
# In units of atm-cm
ApCH2O = 0.0007
ApCH4 = 0.03
ApCO = 0.035
ApHNO2 = 0.0002
ApHNO3 = 0.0005
ApNO = 0.02
ApNO2 = 0.006
ApNO3 = 5e-6
AbO3 = 0.33
ApO3 = 0.0053
ApSO2 = 0.005
AbO2 = 85345.56372007157
AbO4 = 1.2583656276449057e+43
AbBrO = 2.5e-6
AbClNO = 0.00012

qCO2 = 370.0
#ALPHA1 = 0.9111
#ALPHA2 = 1.3529
#OMEGL = 0.8
#GG = 0.7
TAU5 = 1.6
    
amp = 1.0
eps = 0.005

#init_params = np.array([a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, d,
#                        W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3,
#                       ApNO, ApNO2, ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5, amp, eps])
#init_params = np.array([a1, b1, c1, a2, b2, c2, a3, b3, c3, d, eps])
#init_params = np.array([W, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, 
#                        ApNO3, AbO3, ApO3, ApSO2, qCO2, TAU5, amp, eps])
#init_params = np.array([W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, TAU5, amp, eps])
init_params = np.array([a2, b2, c2, a3, b3, c3, d, RH,
                        W, ApHNO2, ApNO2, ApNO3, AbO3, ApO3, ApSO2, 
                        AbO2, TAU5, eps])

print("   initial parameters = ", init_params)


# -- cut last 255 entries from spectrum
#mywav = cube.waves[:-255]
#myblds = nblds[:-255]


# -- define log probability with global wavelengths and spectra arrays
def log_probability(theta):    
    lp = log_prior(theta, mywav, scan)
    if not np.isfinite(lp):
#        print("LOG_PROBABILITY = ", -np.inf)
        return -np.inf
    
    lk = log_likelihood(theta, mywav, myblds, scan)
    if not np.isfinite(lk):
#        print("LOG_LIKELIHOOD = ", -np.inf)
        return -np.inf
    lprb = lp + lk
#    print("LOG_PROBABILITY = ", lprb)
    return lprb


# -- Setting walkers, number of steps, and initial array
nwalkers, ndim, nsteps = 200, init_params.shape[0], 150000
#p0 = init_params * (np.random.rand(nwalkers, ndim)*2)
p0 = init_params * (1 + np.random.randn(nwalkers, ndim)/100.)

# -- Perform MCMC
print("Starting MCMC:")
print("   number of walkers    = ", nwalkers)
print("   number of dimensions = ", ndim)
print("   number of steps      = ", nsteps)
start_time = time.time()

filename = "MCMC_"+scan+"_O2_qel2_bi.h5"
if os.path.isfile(filename):
    backend = emcee.backends.HDFBackend(filename)
    print("MCMC has already been ran to {0} iterations".format(backend.iteration))
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(25) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
#                                        moves = emcee.moves.StretchMove(), backend=backend, pool=pool)
                                        moves=[(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(),0.5),], 
                                        backend=backend, pool=pool)
        nsteps = nsteps - backend.iteration
        sampler.run_mcmc(None, nsteps)
else:
    #Setting up backend to save and monitor progress
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(25) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
#                                        moves = emcee.moves.StretchMove(), backend=backend, pool=pool)
                                        moves=[(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(),0.5),],
                                        backend=backend, pool=pool)
        sampler.run_mcmc(p0, nsteps)

elapsed_time = time.time() - start_time
print("")
print("Total MCMC time = ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print("")
print("   number of walkers    = ", nwalkers)
print("   number of dimensions = ", ndim)
print("   number of steps      = ", nsteps)


# -- Plot walkers
fig, axes = plt.subplots(ndim, sharex=True, figsize=(8,40))
#labels = ['a1', 'b1', 'c1',
#          'a2', 'b3', 'c2',
#          'a3', 'b3', 'c3',
#          'a4', 'b4', 'c4', 'd',        
#          'H2O', 'ApCH2O', 'ApCH4', 'ApCO', 'ApHNO2', 'ApHNO3', 'ApNO', 'ApNO2', 'ApNO3',
#          'AbO3', 'ApO3', 'ApSO2', 'qCO2', 'TAU5', 'amp', 'eps']
#labels = ['a1', 'b1', 'c1',
#          'a2', 'b2', 'c2',
#          'a3', 'b3', 'c3', 'd', 'eps']
#          'a4', 'b4', 'c4', 'd', 'eps']
#labels = ['H2O', 'ApCH2O', 'ApCH4', 'ApCO', 'ApHNO2', 'ApHNO3', 
#          'ApNO', 'ApNO2', 'ApNO3', 'AbO3', 'ApO3', 'ApSO2', 'qCO2', 'TAU5', 'amp', 'eps']
#labels = ['H2O', 'ApHNO2', 'ApNO2', 'ApNO3', 'AbO3', 'ApO3', 'ApSO2', 'TAU5', 'amp', 'eps']
labels = ['a2', 'b2', 'c2', 
          'a3', 'b3', 'c3', 'd', 'RH',
          'H2O', 'ApHNO2', 'ApNO2', 'ApNO3', 'AbO3', 'ApO3', 'ApSO2', 
          'AbO2', 'TAU5', 'eps']

samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i], "k", alpha=0.3)
    ax.set_xlim(0,len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1,0.5)

axes[-1].set_xlabel("step number")
fig.canvas.draw()
fig.savefig("../output/MCMC_walkers_"+scan+"_O2_qel2_bi.png", dpi=300)


# -- Autocorrelation time, burn-in, and flattening
tau = sampler.get_autocorr_time()
burnin = int(3 * np.max(tau))
thin = int(0.5 * np.min(tau))
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
#log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
#log_prior_samples = sampler.get_blobs(discard=burnin, thin=thin, flat=True)

print("Autocorrelation Time = ", tau)
print("burn-in = ", burnin)
print("thin = ", thin)
print("flat chain shape: ", flat_samples.shape)
#print("flat log prob shape: ", log_prob_samples.shape)
#print("flat log prior shape: ", log_prior_samples.shape)

#all_samples = np.concatenate(
#    (flat_samples, log_prob_samples[:,None], log_prior_samples[:,None]), axis=1)

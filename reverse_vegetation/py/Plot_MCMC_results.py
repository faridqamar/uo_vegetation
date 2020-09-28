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
from mcmcFuncs import *
import pysmarts
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


# -- multiplying mean building spectrum by wavelength
print("Multiplying mean building spectrum by wavelength ...")
nblds = blds*cube.waves/1e3
#plt.plot(cube.waves, blds, label='raw spectrum')
#plt.plot(cube.waves, nblds, label='spectrum*wavelength')
#plt.legend()
#plt.show()

mywav = cube.waves[:-200]
myblds = nblds[:-200]

# -- HDF5 mcmc file
filename = "MCMC_"+scan+".h5"
if not os.path.isfile(filename):
    print('Filename {0} does not exist'.format(filename))
else:
    backend = emcee.backends.HDFBackend(filename, read_only=True)
    nwalkers, ndim = 200, 27
    nsteps = backend.iteration
    print("   number of walkers    = ", nwalkers)
    print("   number of dimensions = ", ndim)
    print("   number of steps      = ", nsteps)
    
    # -- Plot walkers
#    print("")
#    print("Plotting Walkers ...")
#    fig, axes = plt.subplots(ndim, sharex=True, figsize=(8,40))
#    labels = ['a1', 'b1', 'c1', 'd1',
#              'a2', 'b3', 'c2', 'd2',
#              'a3', 'b3', 'c3', 'd3',
#              'a4', 'b4', 'c4', 'd4',          
#              'H2O', 'ApCH2O', 'ApHNO2', 'ApHNO3', 'ApNO2', 'ApNO3',
#              'ApO3', 'ApSO2', 'TAU5', 'amp', 'eps']
#    samples = backend.get_chain()
#    for i in range(ndim):
#        ax = axes[i]
#        ax.plot(samples[:,:,i], "k", alpha=0.3)
#        ax.set_xlim(0,len(samples))
#        ax.set_ylabel(labels[i])
#        ax.yaxis.set_label_coords(-0.1,0.5)
#    axes[-1].set_xlabel("step number")
#    fig.canvas.draw()
#    fig.savefig("../output/MCMC_walkers_"+scan+".png", dpi=300)
    
    # -- Calculate tau
    print("Calculating Autocorrelation Time for each parameter...")
    #try:
    #    tau = backend.get_autocorr_time()
    #    print("   Autocorrelation Time = ", tau)
    #    burnin = int(3 * np.max(tau))
    #    thin = int(0.5 * np.min(tau))
    #except:
     #   print("   EXCEPTION RAISED: ")
     #   print("      emcee.autocorr.AutocorrError: ")
     #   print("      The chain is shorter than 50 times the integrated autocorrelation time for 27 parameter(s)")
    burnin = 3000
    thin = 500
    flat_samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
    #log_prob_samples = backend.get_log_prob(discard=burnin, thin=thin, flat=True)
    #log_prior_samples = backend.get_blobs(discard=burnin, thin=thin, flat=True)

    print("   burn-in = ", burnin)
    print("   thin = ", thin)
    print("   flat chain shape: ", flat_samples.shape)
    #print("   flat log prob shape: ", log_prob_samples.shape)
    #print("   flat log prior shape: ", log_prior_samples.shape)
    
    # -- Corner Plot
    print("")
    print("Plotting Corner Plot ...")
    f, ax = plt.subplots(ndim, ndim, figsize=((ndim)*2,(ndim)*2))
    labels = ['a1', 'b1', 'c1', 'd1',
              'a2', 'b3', 'c2', 'd2',
              'a3', 'b3', 'c3', 'd3',
              'a4', 'b4', 'c4', 'd4',          
              'H2O', 'ApCH2O', 'ApHNO2', 'ApHNO3', 'ApNO2', 'ApNO3',
              'ApO3', 'ApSO2', 'TAU5', 'amp', 'eps']
    fig = corner.corner(flat_samples, labels=labels, truths=np.median(flat_samples, axis=0), fig=f)
    f.canvas.draw()
    f.savefig("../output/MCMC_Corner_"+scan+".png", dpi=300)


    # -- Plot a sample of the MCMC solutions
    print("Plotting MCMC Sample Solutions ...")
    fig, ax = plt.subplots(figsize=(10,6))
    inds = np.random.randint(len(flat_samples), size=800)
    for ind in inds:
        sample = flat_samples[ind]
        smrtwav, smrtmod = modelFunc(scan, *sample[:-2])
        maxmod = interpModel(mywav, sample[-2], smrtwav, smrtmod)
        linm, = ax.plot(mywav, maxmod, color='dodgerblue', lw=0.3)
    linb, = ax.plot(mywav, myblds, color='darkred')
    ax.set_xlabel('wavelength [nm]')
    ax.legend([linb, linm], ['data', 'model'])
    fig.savefig("../output/MCMC_models_"+scan+".png", dpi=300)
    

    print("Plotting MCMC Albedo Solutions ...")
    fig, ax = plt.subplots(figsize=(10,6))
    inds = np.random.randint(len(flat_samples), size=800)
    for ind in inds:
        sample = flat_samples[ind]
        albedo = albedoFunc(mywav/1000., *sample[:16])
        linm, = ax.plot(mywav, albedo, color='dodgerblue', lw=0.2)
    ax.set_xlabel('wavelength [nm]')
#    ax.set_ylim(-0.1,0.6)
    fig.savefig("../output/MCMC_albedo_"+scan+"_full.png", dpi=300)
    











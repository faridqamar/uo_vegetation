#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import hyss_util as hu
from plotting import set_defaults

# -- set plotting defaults
set_defaults()

# -- load the data
vspec_avg = np.load("../output/veg_spec_avg.npy")
vspec_sig = np.load("../output/veg_spec_sig.npy")
sspecs    = np.array([np.load("../output/sky_specs/sky_spec_{0:05}.npy" \
                                  .format(i)) for i in range(330)])
waves     = hu.read_header("../data/veg_00000.hdr")["waves"]*1e-3
nwav      = vspec_avg.shape[1]

# -- plot it
nprow = 3
npcol = 3

fig, ax = plt.subplots(nprow,npcol,figsize=(6.5,5),sharex=True,sharey=True)
fig.subplots_adjust(0.1,0.1,0.95,0.95)

for iday in range(nprow*npcol-1):
    iax = iday // npcol
    jax = iday % npcol
    tlo = 2+iday*41
    thi = 2+(iday+1)*41

    for ii in range(10):
        ax[iax,jax].plot((0,nwav),(4*ii-0.5,4*ii-0.5),color="darkorange",
                         lw=0.5)
    ax[iax,jax].imshow(vspec_avg[tlo:thi]/sspecs[tlo:thi],
                       clim=(0.2,0.8),aspect=15)
    ax[iax,jax].set_ylim(thi-tlo,0)

    ax[iax,jax].set_yticks(range(0,44,4))
    ax[iax,jax].set_yticklabels(["{0:02}:{1:02}".format(i,j) for i in 
                                 range(8,19) for j in [0]], fontsize=10)
    ax[iax,jax].grid(0)
    ax[iax,jax].set_xticks([np.argmin(np.abs(waves-i)) for i in 
                            np.arange(0.4,1.2,0.2)])
    ax[iax,jax].set_xticklabels([str(i) for i in np.arange(0.4,1.2,0.2)],
                                fontsize=10)
    ax[iax,jax].grid(0)
ax[2,1].set_xlabel("wavelength [micron]")
ax[2,2].axis("off")
fig.canvas.draw()
#fig.savefig("../output/reflectance_time.eps", clobber=True)
#fig.savefig("../output/reflectance_time.png", clobber=True)

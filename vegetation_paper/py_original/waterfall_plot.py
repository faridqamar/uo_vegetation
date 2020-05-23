#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import hyss_util as hu

# -- utilities
sc_good = np.load("../output/scans_unsaturated.npy")
waves   = hu.read_header("../data/veg_00000.hdr")["waves"]
nspec   = 1000
vspecs  = np.zeros([nspec,waves.size])
sspecs  = np.zeros([nspec,waves.size])


# -- subset the indices for only the close patch
sh    = [1600,1601]
dlabs = np.load("../output/km_00055_ncl10_see314_labs_full.npy")
trind = ((dlabs==2)|(dlabs==5)).reshape(sh)
rr    = (1050,1300)
cr    = (650,1150)

cind, rind = np.meshgrid(np.arange(sh[1]),np.arange(sh[0]))

good = {}

good[1601] = (rind[trind]>=rr[0]) & (rind[trind]<rr[1]) & \
    (cind[trind]>=cr[0]) & (cind[trind]<cr[1])

good[1600] = (rind[:,:1600][trind[:,:1600]]>=rr[0]) & \
    (rind[:,:1600][trind[:,:1600]]<rr[1]) & \
    (cind[:,:1600][trind[:,:1600]]>=cr[0]) & \
    (cind[:,:1600][trind[:,:1600]]<cr[1])


# -- read the vegetation data and sky
for ii,snum in enumerate(sc_good[:nspec]):
    print("\rreading scan number {0}...".format(snum)),
    sys.stdout.flush()

    ncol = hu.read_header("../data/veg_{0}.hdr".format(snum), 
                          verbose=False)["ncol"]

    vspecs[ii] = np.load("../output/veg_specs/veg_specs_{0}.npy"\
                             .format(snum))[:,good[ncol]].mean(1)
    sspecs[ii]  = np.load("../output/sky_specs/sky_spec_{0}.npy"\
                              .format(snum))

# -- perform the fits
slp = np.zeros(nspec)
off = np.zeros(nspec)

for ii in range(nspec):
    print("\rfitting scan number {0}...".format(sc_good[ii])),
    sys.stdout.flush()
    
    slp[ii], off[ii] = np.polyfit(sspecs[ii,:100],vspecs[ii,:100],1)

mod = (slp*sspecs.T+off).T
res = vspecs - mod


# # -- plot it
# plot(waves, (vspecs-mod).T,color='darkred',lw=0.2)

# # -- plot variance of residual before and after 675
# waves = hu.read_header("../data/veg_{0}.hdr".format(snum))["waves"]
# lo = res[:,waves<675.].std(1)
# hi = res[:,waves>=675.].std(1)

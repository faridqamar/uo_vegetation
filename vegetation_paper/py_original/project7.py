#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import hyss_util as hu
import matplotlib.pyplot as plt

# -- get the vegetation pixels
sh    = [1600,1601]
dlabs = np.load("../output/km_00055_ncl10_see314_labs_full.npy")
trind = ((dlabs==2)|(dlabs==5)).reshape(sh)

# -- get the row and column pix
cind, rind = np.meshgrid(np.arange(sh[1]),np.arange(sh[0]))

# -- set range
rr = (1050,1300)
cr = (650,1150)

# -- loop through times and get the appropriate spectra
tind = (trind) & (rind>=rr[0]) & (rind<rr[1]) & (cind>=cr[0]) & (cind<cr[1])
mid_specs = np.zeros((41,848),dtype=float)

for ii in range(0,41):
    cube          = hu.read_hyper("../data/veg_{0:05}.raw".format(ii+2))
    mid_specs[ii] = cube.data[:,tind[:,:cube.ncol]].mean(-1)

# -- get sky specs
sspecs = np.array([np.load("../output/sky_specs/sky_spec_{0:05}.npy" \
                               .format(i+2)) for i in range(42)])

# -- get wavelengths
hdr = hu.read_header("../data/veg_00000.hdr")
waves = hdr["waves"]

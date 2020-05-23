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

# -- get the vegetation as a function of row
dr  = 100
rlo = np.arange(0,sh[0],dr)
rhi = rlo + dr

# -- set the indices for the regions
regs  = [(trind & (rind>=rlo[i]) & (rind<rhi[i])) for i in range(rlo.size)]
nrpix = np.array([reg.sum() for reg in regs])

# -- test on one file
cube  = hu.read_hyper("../data/veg_00055.raw")
specs = np.zeros((rlo.size,cube.nwav),dtype=float)
for ii,reg in enumerate(regs):
    print ii
    if nrpix[ii]>0:
        specs[ii] = cube.data[:,reg].mean(-1)

# -- set the colors and plot
clrs = plt.cm.jet(np.linspace(0.2,0.8,(nrpix>0).sum()))
fig, ax = plt.subplots()
for spec,clr in zip(specs[nrpix>0],clrs):
    ax.plot(cube.waves*1e-3,spec,color=clr)
fig.canvas.draw()

# -- get the sky and plot reflectance
sky = np.load("../output/sky_specs/sky_spec_00055.npy")
clrs = plt.cm.jet(np.linspace(0.2,0.8,(nrpix>0).sum()))
fig, ax = plt.subplots()
for spec,clr in zip(specs[nrpix>0],clrs):
    ax.plot(cube.waves*1e-3,(spec-spec.min())/(sky-sky.min()+1e-3),color=clr)
ax.set_ylim(-0.1,1.1)
fig.canvas.draw()

# -- get out the sky and get the residual
sky = np.load("../output/sky_specs/sky_spec_00055.npy")
clrs = plt.cm.jet(np.linspace(0.2,0.8,(nrpix>0).sum()))
fig, ax = plt.subplots()
for spec,clr in zip(specs[nrpix>0],clrs):
    m, b = np.polyfit(sky,spec,1)
    ax.plot(cube.waves*1e-3,(spec-(m*sky+b))/spec.std(),color=clr)
fig.canvas.draw()


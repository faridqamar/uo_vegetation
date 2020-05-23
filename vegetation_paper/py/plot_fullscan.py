#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import hyss_util as hu


# -- plot the pixelized version
snum  = 360
cube  = hu.read_hyper("../data/veg_{0:05}.raw".format(snum))
waves = hu.read_header("../data/veg_00000.hdr")["waves"]
lam   = [610.,540.,475.]
bands = [np.arange(waves.size)[(waves>=(tlam-10.))&(waves<(tlam+10.))] for 
         tlam in lam]
red   = cube.data[bands[0]].mean(0)
grn   = cube.data[bands[1]].mean(0)
blu   = cube.data[bands[2]].mean(0)
red8  = red*2.**8/2.**12
grn8  = grn*2.**8/2.**12
blu8  = blu*2.**8/2.**12
wr    = red.mean()
wg    = grn.mean()
wb    = blu.mean()
scl   = np.array([wr,wg,wb])
scl  /= scl.max()
scl  /= np.array([0.9,1.0,1.0])
amp   = 2.0
rgb8  = (amp*np.dstack([red8,grn8,blu8])/scl).clip(0,255).astype(np.uint8)

plt.close("all")
fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("Full Resolution Scan (false color)")
im = ax.imshow(rgb8,aspect=0.45,interpolation="nearest")
fig.canvas.draw()
fig.savefig("../output/full_scan_{0:05}.eps".format(snum), clobber=True)
fig.savefig("../output/full_scan_{0:05}.pdf".format(snum), clobber=True)
fig.savefig("../output/full_scan_{0:05}.png".format(snum), clobber=True)


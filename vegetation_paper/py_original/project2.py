#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import hyss_util as hu


# -- utility for plotting
def plot_rgb(hcube,lam=[610.,540.,475.],scl=2.5):
    """
    Plot a false color version of the data cube.
    """
    ind = [np.argmin(np.abs(hcube.waves-clr)) for clr in lam]
    rgb = hcube.data[ind].copy()
    wgt = rgb.mean(0).mean(0)
    scl = scl*wgt[0]/wgt * 2.**8/2.**12

    rgb = (rgb*scl).clip(0,255).astype(np.uint8).transpose(1,2,0)

    fig, ax = plt.subplots(figsize=[6.5,6.5/2])
    fig.subplots_adjust(0.05,0.05,0.95,0.95)
    ax.axis("off")
    im = ax.imshow(rgb,aspect=0.45)
    fig.canvas.draw()

    return


# -- read in the K-Means result
km = pkl.load(open("../output/km_00055_ncl10_seed314.pkl"))


# -- get the file list
flist = sorted(glob.glob("../data/veg_*.raw"))


# -- read in the first file
cube = hu.read_hyper(flist[0])


# -- tag pixels
try:
    data_norm
except:
    print("normalizing full data set...")
    data_norm = 1.0*cube.data.reshape(cube.nwav,cube.nrow*cube.ncol).T.copy()
    data_norm -= data_norm.mean(1,keepdims=True)
    data_norm /= data_norm.std(1,keepdims=True)

    print("predicting labels...")
    dlabs = km.predict(data_norm)


# -- plot tags
clrs = plt.cm.Paired(np.linspace(0,1,km.n_clusters))[:,:3]
tags = (255*clrs[dlabs.reshape(cube.nrow,cube.ncol)]).astype(np.uint8)

fig, ax = plt.subplots(figsize=(6.5,6.5/2))
ax.axis("off")
fig.subplots_adjust(0.05,0.05,0.95,0.95)
im = ax.imshow(tags,aspect=0.45)
fig.canvas.draw()


# -- plot just vegetation
vind  = ((dlabs==2)|(dlabs==5)).reshape(cube.nrow,cube.ncol)
veg   = vind[:,:,np.newaxis]*clrs[2]
vspec = cube.data[:,vind].mean(-1)
plot(cube.waves,vspec)

# -- get the sky
sspec = cube.data[:,:700].mean(-1).mean(-1)


# -- plot the reflectance
refl = (vspec-vspec.min())/(sspec-sspec.min())
plot(cube.waves,refl)

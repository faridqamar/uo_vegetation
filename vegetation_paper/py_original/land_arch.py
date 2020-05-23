#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.cluster import KMeans
import hyss_util as hu

# -- read the rgb image if possible
try:
    rgb8 = np.load("../output/rgb8_00360.npy")
except:
    snum  = 360
    cube  = hu.read_hyper("../data/veg_{0:05}.raw".format(snum))
    waves = hu.read_header("../data/veg_00000.hdr")["waves"]
    lam   = [610., 540., 475.]
    bands = [np.arange(waves.size)[(waves >= (tlam-10.)) & 
                                   (waves < (tlam+ 10.))] for tlam in lam]
    red   = cube.data[bands[0]].mean(0)
    grn   = cube.data[bands[1]].mean(0)
    blu   = cube.data[bands[2]].mean(0)
    red8  = red * 2.**8 / 2.**12
    grn8  = grn * 2.**8 / 2.**12
    blu8  = blu * 2.**8 / 2.**12
    wr    = red.mean()
    wg    = grn.mean()
    wb    = blu.mean()
    scl   = np.array([wr, wg, wb])
    scl  /= scl.max()
    scl  /= np.array([0.9, 1.0, 1.0])
    amp   = 2.0
    rgb8  = (amp * np.dstack([red8, grn8, blu8]) / scl) \
        .clip(0, 255).astype(np.uint8)
    np.save("../output/rgb8_{0:05}.npy".format(snum), rgb8)


# -- plot the rgb image
xs = 16.
ys = 0.5 * xs

fig, ax = plt.subplots(figsize=(xs, ys), dpi=200)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
im = ax.imshow(rgb8, aspect=0.5, interpolation="nearest")
xr = ax.get_xlim()
yr = ax.get_ylim()
txt = ax.text(0.0025 * xr[1], 0.995 * yr[0], "Gregory Dobler (NYU/CUSP)", 
              color="#555555", ha="left", va="bottom", fontsize=18, 
              family="serif")
fig.canvas.draw()
fig.savefig("../output/cusp_uo_hyperspectral_image.png", clobber=True, dpi=200)


# -- plot the vegetation pixels
try:
    veg = np.load("../output/veg_pixels_green_00055.npy")
except:
    km         = pkl.load(open("../output/km_00055_ncl10_seed314.pkl", "rb"))
    cube       = hu.read_hyper("../data/veg_00055.raw")
    data_norm  = 1.0 * cube.data.reshape(cube.nwav, cube.nrow * cube.ncol).T
    data_norm -= data_norm.mean(1, keepdims=True)
    data_norm /= data_norm.std(1, keepdims=True)

    print("predicting labels...")
    clrs  = plt.cm.Paired(np.linspace(0, 1, km.n_clusters))[:, :3]
    dlabs = km.predict(data_norm)
    tags  = (255 * clrs[dlabs.reshape(cube.nrow, cube.ncol)]).astype(np.uint8)
    np.save("../output/tags_00055.npy", tags)

    dlabs_img = dlabs.reshape(cube.nrow, cube.ncol)
    veg       = np.zeros_like(tags)
    for ii in [2,5]:
        veg[dlabs_img==ii] = (255 * clrs[2]).astype(int)

    np.save("../output/veg_pixels_green_00055.npy", veg)


xs = 16.
ys = 0.5 * xs

fig, ax = plt.subplots(figsize=(xs, ys), dpi=200)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
im = ax.imshow(veg, aspect=0.5, interpolation="nearest")
xr = ax.get_xlim()
yr = ax.get_ylim()
txt = ax.text(0.0025 * xr[1], 0.995 * yr[0], "Gregory Dobler (NYU/CUSP)", 
              color="#555555", ha="left", va="bottom", fontsize=18, 
              family="serif")
fig.canvas.draw()
fig.savefig("../output/cusp_uo_vegetation_tags.png", clobber=True, dpi=200)

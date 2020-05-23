#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import hyss_util as hu
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.cluster import KMeans


# -- set the directories
DATADIR = os.path.join("..","data")
OUTDIR  = os.path.join("..","output")


# -- read in the tree pixels
dlabs = np.load("../output/km_00055_ncl10_see314_labs_full.npy")
trind = (dlabs==2)|(dlabs==5)
sh    = [1600,1601]


# -- loop through data cubes
for ii in range(300,330):
    t0    = time.time()
    fname = "veg_{0:05}.raw".format(ii)
    cube  = hu.read_hyper(os.path.join(DATADIR,fname))
    dt    = time.time() - t0
    print("read data cube in {0}m:{1:02}s".format(int(dt//60),int(dt%60)))

    # -- pull off tree pixels (remove last column if necessary)
    vspecs = cube.data[:,trind.reshape(sh)[:,:cube.data.shape[2]]]
    sspec  = cube.data[:,:700].mean(-1).mean(-1)

    # -- write to file
    np.save(os.path.join("..","output","veg_specs",
                         "veg_specs_{0:05}.npy".format(ii)),vspecs)
    np.save(os.path.join("..","output","sky_specs",
                         "sky_spec_{0:05}.npy".format(ii)),sspec)

    dt    = time.time() - t0
    print("processed cube in {0}m:{1:02}s".format(int(dt//60),int(dt%60)))

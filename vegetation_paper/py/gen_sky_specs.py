#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import hyss_util as hu

# -- set the directories
DATADIR = os.path.join("..","data")
OUTDIR  = os.path.join("..","output","sky_specs")


# -- get the file list
flist = sorted(glob.glob(os.path.join(DATADIR,"veg_*.raw")))


# -- set the number of rows to use for the sky
nsrow = 160


# -- loop through data cubes
for tfile in flist:

    # -- get the scan number
    snum = tfile.split("_")[1].replace(".raw","")

    # -- set the output file
    ofile = os.path.join(OUTDIR,"sky_spec_{0}.npy".format(snum))

    # -- check if the sky spectrum exists
    if os.path.isfile(ofile):
        continue

    # -- read in the data file
    t0   = time.time()
    cube = hu.read_hyper(tfile)

    # -- calculate the sky spectrum
    sspec = cube.data[:,:nsrow].mean(-1).mean(-1)

    # -- write to file
    np.save(ofile,sspec)

    # -- alert user
    dt = time.time() - t0
    print("processed cube in {0}m:{1:02}s".format(int(dt//60),int(dt%60)))

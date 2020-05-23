#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The goal here is to identify which scans have clipped sky spectra.
# What we'll do is count the number of pixels equal to 2**12 (clipped)
# for each pixel for each scan and write to a file.

import os
import glob
import numpy as np
import hyss_util as hu

# -- get the file list
DATADIR = "../data"
flist = sorted(glob.glob(os.path.join(DATADIR,"veg*.raw")))


# -- set the output directory
OUTDIR = "../output/clipped"


# -- get the map of number of clips per pixels
cl_val = 2**12 - 1

for fname in flist:
    outname = os.path.join(OUTDIR,os.path.split(fname)[-1] \
                               .replace("veg","nclip").replace("raw","npy"))

    if os.path.isfile(outname):
        continue

    np.save(outname,(hu.read_hyper(fname).data==cl_val).sum(0))

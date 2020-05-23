#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np

# -- utilities
OPATH   = "../output/clipped"
flist   = np.array(sorted(glob.glob(os.path.join(OPATH,"nclip_*.npy"))))
nwav    = 848
wthresh = int(nwav*0.05)
ncol    = 1600 # this is an approximation, ncol may be 1601
nsrow   = int(ncol*0.1)
pthresh = int(ncol*nsrow*0.05)

# -- read in the sky file and clip on fraction of wavelengths and sky pixels 
print("determining unsaturated scans...")
bflags = np.array([(np.load(tfile) > wthresh)[:nsrow].sum() > pthresh for 
                   tfile in flist])

# -- get good scan numbers
print("creating array of good scan numbers...")
sc_good = np.array([i.split("_")[1].replace(".npy","") for i in 
                    flist[~bflags]])

# -- write to file
np.save("../output/scans_unsaturated.npy",sc_good)

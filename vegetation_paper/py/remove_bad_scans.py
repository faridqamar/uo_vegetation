#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import hyss_util as hu

# -- load in the unsaturated scans
good = np.load("../output/scans_unsaturated.npy")

# -- remove bad scans
flist = sorted(glob.glob("../data/veg_*.hdr"))
bad   = [i.split("_")[1].replace(".hdr","") for i in flist if 
         hu.read_header(i,verbose=False)["ncol"] not in [1600,1601]]
good = np.array([i for i in good if i not in bad])

# -- write to file
np.save("../output/good_scans.npy",good)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np
import hyss_util as hu
from multiprocessing import Pool


# -- utilities
DATADIR = os.path.join("..","data")
SKYDIR  = os.path.join("..","output","sky_specs")
SATDIR  = os.path.join("..","output","saturated")
satval  = 2**12 - 1 # the saturation value
nsrow   = 160 # the number of sky rows [10% of image]


# -- define processing pipeline
def process_scan(fname):
    """
    Process a VNIR scan for vegetation.
    """

    # -- get the scan number
    snum = fname.split("_")[1].replace(".raw","")

    # -- set the output files
    skyfile = os.path.join(SKYDIR,"sky_spec_{0}.npy".format(snum))
    satfile = os.path.join(SATDIR,"nsat_{0}.npy".format(snum))

    # -- check if they both exist
    skydone = os.path.isfile(skyfile)
    satdone = os.path.isfile(satfile)
    if skydone and satdone:
        return

    # -- read data file and initialize time
    print("working on scan {0}...".format(snum))
    t0   = time.time()
    cube = hu.read_hyper(fname)

    # -- calculate sky
    if not skydone:
        print("calculating sky...")

        # -- write to file
        np.save(skyfile,cube.data[:,:nsrow].mean(-1).mean(-1))

    # -- calculate number of saturated pixels
    if not satdone:
        print("calculating saturated pixels...")

        # -- write to file
        np.save(satfile,(cube.data==satval).sum(0))

    # -- alert user
    dt = time.time() - t0
    print("processed cube in {0}m:{1:02}s".format(int(dt//60),int(dt%60)))

    return


if __name__=="__main__":

    # -- set number of processors
    nproc = 8

    # -- get the file list
    flist = sorted(glob.glob(os.path.join(DATADIR,"veg_*.raw")))

    # -- define processing runs for parallelization
    def run_processing(fnames):
        for fname in fnames:
            process_scan(fname)

    # -- split the file list into segements for parallelization
    subsz  = int(np.ceil(len(flist)/float(nproc)))
    flists = [flist[i*subsz:(i+1)*subsz] for i in range(nproc)]

    # -- run processing pipeline
    ppool = Pool(nproc)
    ppool.map(run_processing,flists)
        

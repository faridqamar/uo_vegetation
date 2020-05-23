#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Since tagging the individual pixels with the KM clusters from a
# single scan is not particularly accurate, we'll assume that the
# camera pointing is good to a pixel.  Let's see if that's actually
# the case...

import numpy as np
import hyss_util as hu


hdr        = hu.read_header("../data/veg_00000.hdr")
waves      = hdr["waves"] * 1e-3
vspecs     = np.load("../output/veg_specs/veg_specs_00000.npy")
nwav, npix = vspecs.shape
nscan      = 330
sspec      = np.zeros(nwav,dtype=float)
vspec_avg  = np.zeros((nscan,nwav),dtype=float)
vspec_sig  = np.zeros((nscan,nwav),dtype=float)

for ii in range(nscan):
    if (ii+1)%10==0:
        print("working on {0} of {1}...".format(ii+1,nscan))

    vspecs = np.load("../output/veg_specs/"
                     "veg_specs_{0:05}.npy".format(ii))
    sspec[:] = np.load("../output/sky_specs/"
                          "sky_spec_{0:05}.npy".format(ii))

    vspec_avg[ii] = vspecs.mean(1)
    vspec_sig[ii] = vspecs.std(1)

np.save("../output/veg_spec_avg.npy", vspec_avg)
np.save("../output/veg_spec_sig.npy", vspec_sig)
np.save("../output/sky_specs.npy", sspec)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Since tagging the individual pixels with the KM clusters from a
# single scan is not particularly accurate, we'll assume that the
# camera pointing is good to a pixel.  Let's see if that's actually
# the case...

import os
import numpy as np
import matplotlib.pyplot as plt
import hyss_util as hu
from plotting import set_defaults

# -- set the default plotting behavior
set_defaults()


# -- read the first cube
print("reading data cubes...")
cube055 = hu.read_hyper("../data/veg_00055.raw")
cube309 = hu.read_hyper("../data/veg_00309.raw")


# -- get the luminosity images
print("generating luminosity images...")
img_L055 = cube055.data.mean(0)
img_L309 = cube309.data.mean(0)


# -- fit out the offset and amplitude
print("fitting scaling and offset...")
m1, b1 = np.polyfit(img_L055[:,:1600].flatten(),img_L309.flatten(),1)
m2, b2 = np.polyfit(img_L055[:,1:1601].flatten(),img_L309.flatten(),1)

diff1 = (m1*img_L055[:,:1600]+b1) - img_L309
diff2 = (m2*img_L055[:,1:1601]+b2) - img_L309
diff3 = (m2*img_L055[:,2:1601]+b2) - img_L309[:,:1599]


# -- plot it
fig, ax = plt.subplots(3,1,figsize=(6.5,9))
fig.subplots_adjust(0.05,0.05,0.95,0.95)
cr = (14,379)
rr = (1373,1008)
cl = (-100,100)
labs = ("$\Delta x_{pix} = -1$",
        "$\Delta x_{pix} =  0$",
        "$\Delta x_{pix} = +1$")
for tax,img,lab in zip(ax,(diff1,diff2,diff3),labs):
    tax.imshow(img,clim=cl,aspect=0.45)
    tax.set_xlim(cr)
    tax.set_ylim(rr)
    tax.axis("off")
    tax.text(cr[1],rr[1] + 0.025*(rr[1]-rr[0]),lab,ha="right")
ax[0].text(cr[0],rr[1] + 0.025*(rr[1]-rr[0]),
           "Difference of scans separated by 1 week",ha="left")
fig.canvas.draw()
fig.savefig("../output/scan_diff_00055_00309.eps", clobber=True)



# # -- # -- # -- # -- # -- # -- # -- 

# # Now let's check a mean reflectance vs time...

# hdr        = hu.read_header("../data/veg_00000.hdr")
# waves      = hdr["waves"] * 1e-3
# vspecs     = np.load("../output/veg_specs/veg_specs_00000.npy")
# nwav, npix = vspecs.shape
# nscan      = 330
# sspec      = np.zeros(nwav,dtype=float)
# ref_avg    = np.zeros((nscan,nwav),dtype=float)

# for ii in range(nscan):
#     if (ii+1)%10==0:
#         print("working on {0} of {1}...".format(ii+1,nscan))

#     vspecs = np.load("../output/veg_specs/"
#                      "veg_specs_{0:05}.npy".format(ii))
#     sspec[:] = np.load("../output/sky_specs/"
#                           "sky_spec_{0:05}.npy".format(ii))

#     ref_avg[ii] = (vspecs.T/(sspec+1e-3)).mean(0)


# # # -- plot it
# # close("all")
# # fig, ax = plt.subplots()
# # clrs = plt.cm.jet(np.linspace(0,1,43))
# # # for ii in range(nscan):
# # for ii in range(2,43):
# #     ax.plot(waves,ref_avg[ii],color=clrs[ii],lw=0.5)
# # fig.canvas.draw()


# # -- 2D image
# fig, ax = plt.subplots(3,1,figsize=(4.5,6.5),sharex=True)
# fig.subplots_adjust(0.15,0.085,0.95,0.95)
# for ii in range(10):
#     ax[0].plot((0,nwav),(4*ii-0.5,4*ii-0.5),color="darkorange",lw=0.5)
# ax[0].imshow(ref_avg[2:43],clim=(0.2,0.8),aspect=10)

# for ii in range(10):
#     ax[1].plot((0,nwav),(4*ii-0.5,4*ii-0.5),color="darkorange",lw=0.5)
# ax[1].imshow(ref_avg[43:84],clim=(0.2,0.8),aspect=10)

# for ii in range(10):
#     ax[2].plot((0,nwav),(4*ii-0.5,4*ii-0.5),color="darkorange",lw=0.5)
# ax[2].imshow(ref_avg[84:125],clim=(0.2,0.8),aspect=10)

# [i.set_yticks(range(0,44,4)) for i in ax]
# [i.set_yticklabels(["{0:02}:{1:02}".format(i,j) for i in range(8,19) for j in 
#                     [0]]) for i in ax]
# [i.grid(0) for i in ax]
# ax[2].set_xticks([np.argmin(np.abs(waves-i)) for i in np.arange(0.4,1.1,0.1)])
# ax[2].set_xticklabels([str(i) for i in np.arange(0.4,1.1,0.1)])
# ax[2].set_xlabel("wavelength [micron]")
# fig.canvas.draw()

# # -- get NDVI
# ind_ir  = np.argmin(np.abs(waves-0.86))
# ind_vis = np.argmin(np.abs(waves-0.67))
# ndvi    = (ref_avg[:,ind_ir]-ref_avg[:,ind_vis]) / \
#     (ref_avg[:,ind_ir]+ref_avg[:,ind_vis])

# figure()
# plot(ndvi[2:43])
# plot(ndvi[43:84])
# plot(ndvi[84:125])

# ind_ir  = np.argmin(np.abs(waves-0.80))
# ind_vis = np.argmin(np.abs(waves-0.61))
# ndvi    = (ref_avg[:,ind_ir]-ref_avg[:,ind_vis]) / \
#     (ref_avg[:,ind_ir]+ref_avg[:,ind_vis])

# figure()
# plot(ndvi[2:43])
# plot(ndvi[43:84])
# plot(ndvi[84:125])

# ind_ir  = np.argmin(np.abs(waves-0.92))
# ind_vis = np.argmin(np.abs(waves-0.73))
# ndvi    = (ref_avg[:,ind_ir]-ref_avg[:,ind_vis]) / \
#     (ref_avg[:,ind_ir]+ref_avg[:,ind_vis])

# figure()
# plot(ndvi[2:43])
# plot(ndvi[43:84])
# plot(ndvi[84:125])


# # -- integrate over some wavelength range
# ind_ir_lo = np.argmin(np.abs(waves-0.81))
# ind_ir_hi = np.argmin(np.abs(waves-0.91))

# ind_vis_lo = np.argmin(np.abs(waves-0.62))
# ind_vis_hi = np.argmin(np.abs(waves-0.72))

# vis = ref_avg[:,ind_vis_lo:ind_vis_hi].mean(-1)
# ir  = ref_avg[:,ind_ir_lo:ind_ir_hi].mean(-1)

# ndvi    = (ir-vis)/(ir+vis)

# plt.close("all")
# fig, ax = plt.subplots()
# ax.plot(ndvi[2:43])
# ax.plot(ndvi[43:84])
# ax.plot(ndvi[84:125])
# ax.set_xticks(range(0,42,2))
# ax.set_xticklabels(["{0:02}:{1:02}".format(i,j) for i in range(8,19) for j in 
#                     [0,30]],rotation=90)
# fig.canvas.draw()

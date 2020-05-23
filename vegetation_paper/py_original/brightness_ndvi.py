#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Look for variation in buildings spectra normalized 395.46 to 463.93
# micron

import glob
import numpy as np
import hyss_util as hu
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

# -- utilities
waves = hu.read_header("../data/veg_00000.hdr")["waves"]

# -- load the buildings
blds  = np.array([np.load(i) for i in
                  sorted(glob.glob("../output/bld_specs_avg/*.npy"))])
good  = np.array([int(i) for i in np.load("../output/good_scans.npy")]) 
blds  = blds[good]

# -- load the sky
skys = np.array([np.load(i) for i in 
                 sorted(glob.glob("../output/sky_specs/*.npy"))])
skys = skys[good]

# -- normalize spectra
ms, bs = [], []
for ii in range(blds.shape[0]):
    m, b = np.polyfit(blds[ii,:100],blds[0,:100],1)
    ms.append(m)
    bs.append(b)

ms   = np.array(ms)
bs   = np.array(bs)
norm = blds*ms[:,np.newaxis] + bs[:,np.newaxis]
rat  = norm/norm[0]

# -- get vegetation spectra
vegs = np.load("../output/veg_patch_specs.npy")
ss, os = [], []
for ii in range(vegs.shape[0]):
    s, o = np.polyfit(vegs[ii,:100],vegs[0,:100],1)
    ss.append(s)
    os.append(o)

ss    = np.array(ms)
os    = np.array(bs)
vnorm = vegs*ss[:,np.newaxis] + os[:,np.newaxis]
vrat  = vnorm/vnorm[0]

# -- get brightnesses
norm = vrat/rat
brgt = norm.mean(1)

# -- estimate ndvi
ref    = (vegs - vegs.min(1,keepdims=True))/(skys-skys.min(1,keepdims=True))
ind_ir = np.argmin(np.abs(waves-860.))
ind_vs = np.argmin(np.abs(waves-670.))
ndvi   = (ref[:,ind_ir]-ref[:,ind_vs])/(ref[:,ind_ir]+ref[:,ind_vs])

# -- outlier rejection
ind = brgt < 2.0

# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
sc_sub = sc[sc.filename.isin(["veg_{0:05}.raw".format(i) for i in good])]

temps = sc_sub.temperature
humid = sc_sub.humidity
pm25  = sc_sub.pm25
o3   = sc_sub.o3

# -- some plots
dark_plot()
close("all")

figure()
plot(ndvi[ind],brgt[ind],'o',mew=0,ms=2,color="dodgerblue")
plot(ndvi[ndvi<0],brgt[ndvi<0],'o',mew=0,ms=4,color="darkred")
xlabel("NDVI")
ylabel("\"brightness\"")
savefig("../output/brightness_vs_ndvi.png", clobber=True, facecolor="k")

figure()
avg = np.median(norm[humid.values<99.],0)
sig = norm[humid.values<99.].std(0)
fill_between(waves,avg-sig,avg+sig,color="darkgoldenrod",alpha=0.5)
#plot(waves,norm[humid.values<99.][::10].T,lw=0.2,color="indianred")
plot(waves,avg,color="darkred",lw=3)
xlabel("wavelength [nm]")
ylabel("\"brightness\"")
xlim(400,1000)
ylim(0.6,2.0)
savefig("../output/brightness_median_stddev.png", clobber=True, facecolor="k")

figure()
avg = np.median(norm[humid.values<99.],0)
plot(waves,avg,color="darkred",lw=3)
plot(waves,norm[humid.values<99.][::10].T,lw=0.2,color="indianred")
xlabel("wavelength [nm]")
ylabel("\"brightness\"")
xlim(400,1000)
ylim(0.6,2.0)
savefig("../output/brightness_median_ex.png", clobber=True, facecolor="k")

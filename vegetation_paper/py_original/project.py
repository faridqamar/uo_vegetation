#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import hyss_util as hu
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.cluster import KMeans

# -- clear all plotting and reset defaults
set_default = True
plt.close("all")

if set_default:
    plt.rcParams["axes.facecolor"]  = "w"
    plt.rcParams["axes.grid"]       = True
    plt.rcParams["axes.axisbelow"]  = True
    plt.rcParams["axes.linewidth"]  = 0
    plt.rcParams["axes.labelcolor"] = "k"

    plt.rcParams["figure.facecolor"]      = "w"
    plt.rcParams["figure.subplot.bottom"] = 0.125
    plt.rcParams["figure.subplot.left"]   = 0.1

    plt.rcParams["lines.linewidth"] = 2

    plt.rcParams["grid.color"]     = "#444444"
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.linestyle"] = ":"

    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.color"]     = "k"
    plt.rcParams["ytick.color"]     = "k"

    plt.rcParams["text.color"] = "k"


# -- set the directories
DATADIR = os.path.join("..","data")
OUTDIR  = os.path.join("..","output")


# -- read in a scan
try:
    cube
except:
    fname = "veg_00055.raw"
    hdr   = hu.read_header(os.path.join(DATADIR,fname.replace("raw","hdr")))
    cube  = hu.read_hyper(os.path.join(DATADIR,fname))


# -- subset the data for clustering
print("subsampling data cube...")
samp = 16
dsub = cube.data[:,::16,::16].copy().astype(float)


# -- plot the pixelized version
print("plotting RGB...")
def make_rgb8(dcube,lam=[610.,540.,475.],scl=2.5):
    ind = [np.argmin(np.abs(hdr["waves"]-clr)) for clr in lam]
    rgb = dcube[ind]
    wgt = rgb.mean(0).mean(0)
    scl = scl*wgt[0]/wgt * 2.**8/2.**12

    return (rgb*scl).clip(0,255).astype(np.uint8).transpose(1,2,0)

rgb_full = make_rgb8(cube.data,scl=1.5)
fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("Full Resolution Scan (false color)")
im = ax.imshow(rgb_full,aspect=0.45)
fig.canvas.draw()
fig.savefig("../output/full_scan_55.eps", clobber=True)


rgb     = make_rgb8(dsub,scl=1.5)
fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("16x Spatial Sub-sampling")
im = ax.imshow(rgb,aspect=0.45)
fig.canvas.draw()
fig.savefig("../output/sub_scan_55.eps", clobber=True)



# -- cluster subset and save
print("normalizing data subset...")
dsub_norm  = dsub.reshape(dsub.shape[0],dsub.shape[1]*dsub.shape[2]).T.copy()
dsub_norm -= dsub_norm.mean(1,keepdims=True)
dsub_norm /= dsub_norm.std(1,keepdims=True)

try:
    km
except:
    print("K-Means clustering Npnts={0},Nfeat={1}...".format(*dsub_norm.shape))
    n_clusters   = 10
    n_jobs       = 16
    random_state = 314
    km           = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, 
                          random_state=random_state)
    km.fit(dsub_norm)
    pkl.dump(km,open("../output/km_00055_ncl10_seed314.pkl","wb"))


# -- plot the clusters
print("plotting clusters...")
clrs    = plt.cm.Paired(np.linspace(0,1,n_clusters))[:,:3]
fig, ax = plt.subplots(2,5,figsize=[6.5,6.5*4/5/3],sharex=True,sharey=True)
fig.subplots_adjust(0.05,0.3,0.95,0.95)
axs = ax.flatten()
for ii in range(n_clusters):
    axs[ii].set_yticklabels("")
    axs[ii].set_xticklabels("")
    axs[ii].plot(cube.waves*1e-3,km.cluster_centers_[ii],color=clrs[ii])
    axs[ii].set_xlim([0.4,1.0])
    axs[ii].set_ylim([-1.5,2.5])
    axs[ii].set_xticks([0.5,0.7,0.9])
    axs[ii].set_xticklabels([str(i) for i in axs[ii].get_xticks()])
    xlim, ylim = axs[ii].get_xlim(), axs[ii].get_ylim()
    axs[ii].text(xlim[1],ylim[1],str(ii),ha="right",va="top")
    axs[ii].grid(0)
fig.text(0.5,0.04,"wavelength [micron]",ha="center")
fig.canvas.draw()
fig.savefig("../output/kmeans_55_ncl10.eps", clobber=True)



# -- plot the K-Means tags at low resolution
print("creating tagged image...")
tags_lo = np.zeros_like(rgb)
sh_lo   = tags_lo.shape
for ii in range(n_clusters):
    tags_lo[(km.labels_==ii).reshape(sh_lo[:2])] = (255*clrs[ii]).astype(int)

fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("K-Means cluster tags (low resolution)")
im = ax.imshow(tags_lo,aspect=0.45)
fig.canvas.draw()
fig.savefig("../output/cluster_tags_sub.eps", clobber=True)



# -- tag the high resolution image and plot
try:
    data_norm
except:
    print("normalizing full data set...")
    data_norm = 1.0*cube.data.reshape(cube.nwav,cube.nrow*cube.ncol).T.copy()
    data_norm -= data_norm.mean(1,keepdims=True)
    data_norm /= data_norm.std(1,keepdims=True)

    print("predicting labels...")
    dlabs = km.predict(data_norm)
    tags  = (255*clrs[dlabs.reshape(cube.nrow,cube.ncol)]).astype(np.uint8)

print("plotting tags...")
fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("K-Means cluster tags (full resolution)")
im = ax.imshow(tags,aspect=0.45)
fig.canvas.draw()
fig.savefig("../output/cluster_tags.eps", clobber=True)



# -- plot just the trees
print("showing just vegetation...")
dlabs_img = dlabs.reshape(cube.nrow,cube.ncol)
veg       = np.zeros_like(tags)
for ii in [2,5]:
    veg[dlabs_img==ii] = (255*clrs[2]).astype(int)

fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
ax.axis("off")
ax.set_title("vegetation pixels")
im = ax.imshow(veg,aspect=0.45)
fig.canvas.draw()
fig.savefig("../output/vegetation_pixels.eps", clobber=True)



# -- plot the mean vegetation spectrum
print("plotting mean vegetation spectrum...")
veg_sp = data_norm[(dlabs==2)|(dlabs==5)].mean(0)

fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.12,0.18,0.95,0.9)
ax.plot(cube.waves*1e-3,veg_sp,c="lime")
ax.set_xlabel("wavelength [micron]")
ax.set_ylabel("intensity [arb units]")
ax.set_xlim([0.4,1.0])
xlim, ylim = ax.get_xlim(), ax.get_ylim()
ax.text(xlim[1],ylim[1]+0.03*(ylim[1]-ylim[0]),"mean vegetation spectrum",
        ha="right")
fig.canvas.draw()
fig.savefig("../output/vegetation_spectrum.eps", clobber=True)



# -- get the mean sky spectrum
print("plotting mean sky spectrum...")
skyline = 700
sind    = (np.arange(cube.nrow*cube.ncol) // cube.ncol) < skyline
sky_sp  = data_norm[sind].mean(0)
sky_sp -= sky_sp.min()
sky_sp /= sky_sp.max()

fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.12,0.18,0.95,0.9)
ax.plot(cube.waves*1e-3,sky_sp,color="dodgerblue")
ax.set_xlabel("wavelength [micron]")
ax.set_ylabel("intensity [arb units]")
ax.set_xlim([0.4,1.0])
ax.set_ylim([-0.1,1.1])
xlim, ylim = ax.get_xlim(), ax.get_ylim()
ax.text(xlim[1],ylim[1]+0.03*(ylim[1]-ylim[0]),"mean normalized sky spectrum",
        ha="right")
fig.canvas.draw()
fig.savefig("../output/sky_spectrum.eps", clobber=True)



# -- plot the reflectance
print("plotting vegetation reflectance...")
vspt    = cube.data[:,((dlabs==2)|(dlabs==5)) \
                          .reshape(cube.nrow,cube.ncol)].mean(-1)
sspt    = cube.data[:,:700].mean(-1).mean(-1)
reflect = (vspt - vspt.min())/(sspt - sspt.min())

fig, ax = plt.subplots(figsize=[6.5,6.5/2])
fig.subplots_adjust(0.12,0.18,0.95,0.9)
ax.plot(cube.waves*1e-3,reflect)
ax.set_xlabel("wavelength [micron]")
ax.set_ylabel("reflectance")
ax.set_xlim([0.4,1.0])
ax.set_ylim([0.0,1.0])
xlim, ylim = ax.get_xlim(), ax.get_ylim()
fig.canvas.draw()
fig.savefig("../output/reflectance.eps", clobber=True)





## -- plot the reflectance
#print("plotting vegetation reflectance...")
#vspt = cube.data[:,((dlabs==2)|(dlabs==5)) \
#                       .reshape(cube.nrow,cube.ncol)].mean(-1)
#
#sspt = cube.data[:,:700].mean(-1).mean(-1)
#
#reflect = (vspt - vspt.min())/(sspt - sspt.min())
#pos     = (vspt + sig - vspt.min())/(sspt - sspt.min())
#neg     = (vspt - sig - vspt.min())/(sspt - sspt.min())
#
#fig, ax = plt.subplots(figsize=[6.5,6.5/2])
#fig.subplots_adjust(0.12,0.18,0.95,0.9)
#ax.plot(cube.waves*1e-3,reflect)
#ax.plot(cube.waves*1e-3,pos, c='darkgoldenrod')
#ax.plot(cube.waves*1e-3,neg, c='darkgoldenrod')
#ax.set_xlabel("wavelength [micron]")
#ax.set_ylabel("reflectance")
#ax.set_xlim([0.4,1.0])
#ax.set_ylim([0.0,1.0])
#xlim, ylim = ax.get_xlim(), ax.get_ylim()
#fig.canvas.draw()
#plt.show()
#plt.pause(1e-3)


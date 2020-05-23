#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Look for variation in buildings normalized **building** spectra
# 395.46 to 463.93 micron

import sys
import glob
import numpy as np
import hyss_util as hu
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

# -- utilities
waves = hu.read_header("../data/veg_00000.hdr")["waves"]
nwav  = waves.size
nrow  = 44
ncol  = 157
npix  = nrow*ncol
seed  = 3141
inds  = np.argsort(np.random.rand(npix))
ind0  = inds[:inds.size//2]
ind1  = inds[inds.size//2:]
bldsp = np.zeros([nwav,nrow,ncol],dtype=np.uint16)
flist = sorted([i for i in glob.glob("../output/bld_specs/bld_specs_*.npy")
                if "avg" not in i])
blds0 = np.zeros([len(flist),nwav],dtype=float)
blds1 = np.zeros_like(blds0)
good  = np.array([int(i) for i in np.load("../output/good_scans.npy")]) 

# -- load the buildings
for ii, fname in enumerate(flist):
    if (ii+1)%10==0:
        print("\r{0} of {1}...".format(ii+1,len(flist))),
        sys.stdout.flush()

    blds0[ii,:] = np.load(fname).reshape(nwav,nrow*ncol)[:,ind0].mean(-1)
    blds1[ii,:] = np.load(fname).reshape(nwav,nrow*ncol)[:,ind1].mean(-1)


# -- read in the building specs and create 2 subsets
blds0 = blds0[good]
blds1 = blds1[good]

# -- normalize spectra
ms, bs = [], []
for ii in range(blds0.shape[0]):
    m, b = np.polyfit(blds0[ii,:100],blds0[0,:100],1)
    ms.append(m)
    bs.append(b)

ms   = np.array(ms)
bs   = np.array(bs)
norm = blds0*ms[:,np.newaxis] + bs[:,np.newaxis]
rat  = norm/norm[0]

# -- get vegetation spectra
ss, os = [], []
for ii in range(blds1.shape[0]):
    s, o = np.polyfit(blds1[ii,:100],blds1[0,:100],1)
    ss.append(s)
    os.append(o)

ss    = np.array(ms)
os    = np.array(bs)
vnorm = blds1*ss[:,np.newaxis] + os[:,np.newaxis]
vrat  = vnorm/vnorm[0]

# -- PCA
pca = PCA(n_components=6)
pca.fit(vrat/rat)
pamps = pca.transform(vrat/rat)

# -- Factor Analysis
fan = FactorAnalysis(n_components=6)
fan.fit(vrat/rat)
famps = fan.transform(vrat/rat)

# -- ICA
ica = FastICA(n_components=6)
ica.fit(vrat/rat)
iamps = ica.transform(vrat/rat)

# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
sc_sub = sc[sc.filename.isin(["veg_{0:05}.raw".format(i) for i in good])]

temps = sc_sub.temperature
humid = sc_sub.humidity
pm25  = sc_sub.pm25
o3   = sc_sub.o3

# -- make plots
def plot4(amps, c0=0, c1=1):
    dark_plot()
    close("all")

    figure(figsize=(10,10))

    try:
        fndvi = ndvi.clip(0.4,0.8)
        clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())
        gcf().add_subplot(221); scatter(amps[:,1],amps[:,3],c=clrs,s=10,
                                        linewidths=0)
        subplots_adjust(0.1)
        ylabel("PCA component {0}".format(c1))
        title("NDVI")
    except:
        pass

    clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
    gcf().add_subplot(222); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    title("Humidity")

    clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
    gcf().add_subplot(223); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    xlabel("PCA component {0}".format(c0))
    ylabel("PCA component {0}".format(c1))
    title("PM2.5")

    clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())
    gcf().add_subplot(224); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    xlabel("PCA component {0}".format(c0))
    title("O3")



close("all")
for ii in range (5):
    clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
    figure(figsize=[7,5]); scatter(o3,amps[:,ii],c=clrs,s=10,linewidths=0)
    subplots_adjust(0.15)
    xlabel("O3 [ppm]")
    ylabel("PCA component {0}".format(ii))
    title("Humidity")




def ploto3(amps):
    close("all")
    figure(figsize=[10,10])
    for ii in range (6):
        clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
        gcf().add_subplot(3,2,ii+1)
        scatter(o3,amps[:,ii],c=clrs,s=10,linewidths=0)
        subplots_adjust(0.15)
        xlabel("O3 [ppm]")
        ylabel("PCA component {0}".format(ii))
        title("Humidity")
    plt.subplots_adjust(0.05,0.05,0.95,0.95)


def plotpm25(amps):
    close("all")
    figure(figsize=[10,10])
    for ii in range (6):
        clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
        gcf().add_subplot(3,2,ii+1)
        scatter(pm25,amps[:,ii],c=clrs,s=10,linewidths=0)
        subplots_adjust(0.15)
        xlabel("PM2.5 [ppm]")
        ylabel("PCA component {0}".format(ii))
        title("Humidity")
    plt.subplots_adjust(0.05,0.05,0.95,0.95)


def plottemp(amps):
    close("all")
    figure(figsize=[10,10])
    for ii in range (6):
        clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
        gcf().add_subplot(3,2,ii+1)
        scatter(temps,amps[:,ii],c=clrs,s=10,linewidths=0)
        subplots_adjust(0.15)
        xlabel("T [K]")
        ylabel("PCA component {0}".format(ii))
        title("Humidity")
    plt.subplots_adjust(0.05,0.05,0.95,0.95)




# -- let's try to find some weights that make a nice correlation
data = vrat/rat
data = data[1:]
o3m  = o3[1:]

data -= data.mean(0)
data /= data.std(0)
o3m  -= o3m.mean()
o3m  /= o3m.std()

dTd    = np.dot(data.T,data)
o3mdT  = np.dot(data.T,o3m)
dTdinv = np.linalg.pinv(dTd)
wgto3  = np.dot(o3mdT,dTdinv)


pm25m  = pm25[1:]

pm25m  -= pm25m.mean()
pm25m  /= pm25m.std()

pm25mdT  = np.dot(data.T,pm25m)
wgtpm25  = np.dot(pm25mdT,dTdinv)

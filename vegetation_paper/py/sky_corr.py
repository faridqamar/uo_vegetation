#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Calculate the differences between sky from one scan to another.

import sys
import glob
import numpy as np
import pandas as pd
import hyss_util as hu
from sklearn.decomposition import PCA

# -- load the sky spectra and select the good scans
flist = sorted(glob.glob("../output/sky_specs/*.npy"))
skys  = np.array([np.load(i) for i in flist])
good  = np.array([int(i) for i in np.load("../output/good_scans.npy")])
skys  = skys[good]

# -- get wavelengths
waves = hu.read_header("../data/veg_00000.hdr")["waves"]

# -- loop (yuck!) through
nsky = skys.shape[0]
diff = np.zeros([nsky,nsky])
for ii in range(nsky):
    if (ii+1)%100==0:
        print("\r{0} of {1}".format(ii+1,nsky)),
        sys.stdout.flush()
    for jj in range(ii,nsky):
        diff[ii,jj] = ((skys[ii]-skys[jj])**2).sum() / \
            ((skys[ii]+skys[jj])**2).sum()

for ii in range(nsky):
    for jj in range(ii):
        diff[ii,jj] = diff[jj,ii]

for ii in range(nsky):
    diff[ii,ii] = 1.0

# -- pull off the pairs of indices that represent close sky
cind, rind = np.meshgrid(np.arange(nsky),np.arange(nsky))

thresh = 0.001
closer = rind[diff<thresh]
closec = cind[diff<thresh]

mind  = np.bincount(closer).argmax()
#snums = good[closec[closer==mind]]
snums = good
plt.plot(waves,skys[mind],lw=3,color='k')
plt.plot(waves,skys[closec[closer==mind]].T,lw=0.1,color='darkred')


# -- get the vegetation spectra for the first set
sh    = [1600,1601]
dlabs = np.load("../output/km_00055_ncl10_see314_labs_full.npy")
trind = ((dlabs==2)|(dlabs==5)).reshape(sh)
rr    = (1050,1300)
cr    = (650,1150)

vcind, vrind = np.meshgrid(np.arange(sh[1]),np.arange(sh[0]))

vgood = {}

vgood[1601] = (vrind[trind]>=rr[0]) & (vrind[trind]<rr[1]) & \
    (vcind[trind]>=cr[0]) & (vcind[trind]<cr[1])

vgood[1600] = (vrind[:,:1600][trind[:,:1600]]>=rr[0]) & \
    (vrind[:,:1600][trind[:,:1600]]<rr[1]) & \
    (vcind[:,:1600][trind[:,:1600]]>=cr[0]) & \
    (vcind[:,:1600][trind[:,:1600]]<cr[1])


nvegs = len(snums)
try:
    vegs = np.load("../output/veg_patch_specs.npy")
except:

    vegs = np.zeros([nvegs,skys.shape[1]])

    for ii,snum in enumerate(snums):
        print("\rreading scan number {0} or {1}...".format(ii+1,len(snums))),
        sys.stdout.flush()

        ncol = hu.read_header("../data/veg_{0:05}.hdr".format(snum),
                              verbose=False)["ncol"]

        vegs[ii] = np.load("../output/veg_specs/veg_specs_{0:05}.npy" \
                               .format(snum))[:,vgood[ncol]].mean(1)

    np.save("../output/veg_patch_specs.npy",vegs)


# -- get the residuals
#gskys = skys[closec[closer==mind]].copy()
gskys = skys.copy()
ress  = np.zeros_like(vegs)

for ii in range(vegs.shape[0]):
    m, b = np.polyfit(gskys[ii,:100],vegs[ii,:100],1)
    ress[ii] = vegs[ii] - (m*gskys[ii]+b)


# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
sc_sub = sc[sc.filename.isin(["veg_{0:05}.raw".format(i) for i in snums])]

temps = sc_sub.temperature
humid = sc_sub.humidity
pm25  = sc_sub.pm25
o3   = sc_sub.o3
rivi  = (ress[:,400:600]-ress[:,np.newaxis,400]).sum(1)

ref     = (vegs - vegs.min(1,keepdims=True))/(gskys-gskys.min(1,keepdims=True))
ind_ir  = np.argmin(np.abs(waves-860.))
ind_vis = np.argmin(np.abs(waves-670.))
ndvi    = (ref[:,ind_ir]-ref[:,ind_vis]) / \
    (ref[:,ind_ir]+ref[:,ind_vis])




# -- plot these vals vs the height of first peak
figure()
plot(temps,ress[:,230],'o')
figure()
plot(humid,ress[:,230],'o')
figure()
plot(pm25,ress[:,230],'o')

figure()
plot(temps,rivi,'o')
figure()
plot(humid,rivi,'o')
figure()
plot(pm25,rivi,'o')


figure()
plot(temps,ress[:,495]-ress[:,230],'o')
figure()
plot(humid,ress[:,495]-ress[:,230],'o')
figure()
plot(pm25,ress[:,495]-ress[:,230],'o')
figure()
plot(o3,ress[:,495]-ress[:,230],'o')


# -- plot vs pm2.5
clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())

for ii in range(len(snums)):
    plot(waves,ress[ii],lw=0.2,color=clrs[ii])


# -- get PCA components of residuals
norm = (ress - ress.mean(1,keepdims=True))/ress.std(1,keepdims=True)
pca = PCA(n_components=5)
pca.fit(norm)
amps = pca.transform(ress)



pca = PCA(n_components=5)
pca.fit(ress)
amps = pca.transform(ress)


fndvi = ndvi.clip(0.4,0.8)
clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())


pcaclrs = plt.cm.jet((amps[:,1]-amps[:,1].min())/(amps[:,1]-amps[:,1].min()) \
                         .max())


fndvi = ndvi.clip(0.4,0.8)
clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())
figure(figsize=[7,5]); scatter(amps[:,0],amps[:,2],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component 0")
ylabel("PCA component 2")
title("NDVI")

clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
figure(); scatter(amps[:,0],amps[:,2],c=clrs,s=10,linewidths=0)
title("Humidity")

clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
figure(); scatter(amps[:,0],amps[:,2],c=clrs,s=10,linewidths=0)
title("PM2.5")

clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())
figure(); scatter(amps[:,0],amps[:,2],c=clrs,s=10,linewidths=0)
title("O3")


dark_plot()
close("all")

fndvi = ndvi.clip(0.4,0.8)
clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())
figure(figsize=[7,5]); scatter(amps[:,1],amps[:,3],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component 1")
ylabel("PCA component 3")
title("NDVI")

clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
figure(figsize=[7,5]); scatter(amps[:,1],amps[:,3],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component 1")
ylabel("PCA component 3")
title("Humidity")

# clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
# #clrs[:,3] = 0.9*(o3-o3.min())/(o3-o3.min()).max()+0.1
# szs = 5+30*(o3-o3.min())/(o3-o3.min()).max()
# figure(figsize=[7,5]); scatter(amps[:,1],amps[:,3],c=clrs,s=szs,
#                                linewidths=0,alpha=0.8)
# xlabel("PCA component 1")
# ylabel("PCA component 3")
# title("Humidity")

clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
figure(figsize=[7,5]); scatter(amps[:,1],amps[:,3],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component 1")
ylabel("PCA component 3")
title("PM2.5")

clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())
figure(figsize=[7,5]); scatter(amps[:,1],amps[:,3],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component 1")
ylabel("PCA component 3")
title("O3")


w10 = ((humid>=5.)&(humid<15.)).values
w30 = ((humid>=25.)&(humid<35.)).values
w50 = ((humid>=45.)&(humid<55.)).values
w70 = ((humid>=65.)&(humid<75.)).values
w90 = ((humid>=85.)&(humid<95.)).values

figure()
#plot(o3[w10],amps[w10,1],'o')
plot(o3[w30],amps[w30,1],'o')
plot(o3[w50],amps[w50,1],'o')
plot(o3[w70],amps[w70,1],'o')
plot(o3[w90],amps[w90,1],'o')

figure()
#plot(o3[w10],amps[w10,3],'o')
plot(o3[w30],amps[w30,3],'o')
plot(o3[w50],amps[w50,3],'o')
plot(o3[w70],amps[w70,3],'o')
plot(o3[w90],amps[w90,3],'o')




# -- make plots
def plot4(c0=0, c1 = 1):
    dark_plot()
    close("all")
    
    figure(figsize=(10,10))
    
    fndvi = ndvi.clip(0.4,0.8)
    clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())
    gcf().add_subplot(221); scatter(amps[:,1],amps[:,3],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    ylabel("PCA component {0}".format(c1))
    title("NDVI")
    
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


close("all")
figure()
for ii in range(5):
    gcf().add_subplot(5,1,ii+1)
    plot(waves,pca.components_[ii])




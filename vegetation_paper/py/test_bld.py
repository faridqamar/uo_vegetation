#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import hyss_util as hu
from sklearn.decomposition import PCA

# -- utitlities
waves = hu.read_header("../data/veg_00000.hdr")["waves"]

# -- read skies
print("reading skies...")
try:
    skys
except:
    skys = np.array([np.load(i) for i in 
                     sorted(glob.glob("../output/sky_specs/*.npy"))])

# -- read buildings
print("reading blds...")
blds = np.array([np.load(i) for i in 
                 sorted(glob.glob("../output/bld_specs/*.npy"))])

# -- regress
print("regressing...")
ress = np.zeros_like(blds)

for ii in range(ress.shape[0]):
    m, b = np.polyfit(skys[ii,:100],blds[ii,:100],1)
    ress[ii] = blds[ii] - (m*skys[ii] + b)

# -- select good scans
good  = np.array([int(i) for i in np.load("../output/good_scans.npy")])
skys = skys[good]
ress = ress[good]

# -- run PCA
pca = PCA(n_components=5)
pca.fit(ress)
amps = pca.transform(ress)

# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
sc_sub = sc.iloc[good]

temps = sc_sub.temperature
humid = sc_sub.humidity
pm25  = sc_sub.pm25
o3   = sc_sub.o3

# -- make plots
#c0, c1 = 0, 1
#c0, c1 = 0, 3
#c0, c1 = 1, 2
#c0, c1 = 1, 3
#c0, c1 = 2, 3
#c0, c1 = 3, 4
c0, c1 = 0, 1
dark_plot()
close("all")

clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
figure(figsize=[7,5]); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component {0}".format(c0))
ylabel("PCA component {0}".format(c1))
title("Humidity")

clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
figure(figsize=[7,5]); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component {0}".format(c0))
ylabel("PCA component {0}".format(c1))
title("PM2.5")

clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())
figure(figsize=[7,5]); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("PCA component {0}".format(c0))
ylabel("PCA component {0}".format(c1))
title("O3")



c0, c1 = 0, 1

close("all")
clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
figure(figsize=[7,5]); scatter(o3,amps[:,c0],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("O3 [ppm]")
ylabel("PCA component {0}".format(c0))
title("Humidity")
figure(figsize=[7,5]); scatter(o3,amps[:,c1],c=clrs,s=10,linewidths=0)
subplots_adjust(0.15)
xlabel("O3 [ppm]")
ylabel("PCA component {0}".format(c1))
title("Humidity")



close("all")
figure()
for ii in range(5):
    gcf().add_subplot(5,1,ii+1)
    plot(waves,pca.components_[ii])




# clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
# figure(); scatter(o3,amps[:,0],c=clrs,s=10,linewidths=0,alpha=0.5)
# figure(); scatter(o3,amps[:,1],c=clrs,s=10,linewidths=0,alpha=0.5)
# figure(); scatter(o3,amps[:,2],c=clrs,s=10,linewidths=0,alpha=0.5)
# figure(); scatter(o3,amps[:,3],c=clrs,s=10,linewidths=0,alpha=0.5)
# figure(); scatter(o3,amps[:,4],c=clrs,s=10,linewidths=0,alpha=0.5)
# title("Humidity")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Look for variation in buildings spectra normalized 395.46 to 463.93
# micron

import glob
import numpy as np
import hyss_util as hu
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

# -- utilities
dark_plot()
waves = hu.read_header("../data/veg_00000.hdr")["waves"]

# -- load the buildings
blds  = np.array([np.load(i) for i in
                  sorted(glob.glob("../output/bld_specs/bld_specs_avg_*.npy"))
                  ])
good  = np.array([int(i) for i in np.load("../output/good_scans.npy")]) 
blds  = blds[good]

# -- normalize spectra
ms, bs = [], []
for ii in range(blds.shape[0]):
    m, b = np.polyfit(blds[ii,-100:],blds[0,-100:],1)
    ms.append(m)
    bs.append(b)

ms   = np.array(ms)
bs   = np.array(bs)
norm = blds*ms[:,np.newaxis] + bs[:,np.newaxis]
rat  = norm/norm[0]

# -- get vegetation spectra
vegs = np.array([np.load(i) for i in 
                 glob.glob("../output/alt_specs/alt_specs_avg*.npy")])
vegs = vegs[good]
ss, os = [], []
for ii in range(vegs.shape[0]):
    s, o = np.polyfit(vegs[ii,-100:],vegs[0,-100:],1)
    ss.append(s)
    os.append(o)

ss    = np.array(ss)
os    = np.array(os)
vnorm = vegs*ss[:,np.newaxis] + os[:,np.newaxis]
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





# -- more plots:
dark_plot()
figure()
plot(o3*10000)
imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld
grid(0)

figure()
plot(temps*18-800)
imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld

figure()
plot(humid*8)
imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld


rrat = ((vrat/rat).T).clip(0.6,2.0)
rgb  = rrat - rrat.min()
rgb /= rgb.max()
rgb  = np.dstack([255*rgb for i in range(3)])

o3norm  = humid.values-humid.values.min()
o3norm /= o3norm.max()
o3norm *= 255

o3rgb = np.zeros_like(rgb)

for ii in range(rgb.shape[1]):
    for jj in range(3):
        o3rgb[:,ii,jj] = o3norm[ii]


# -- multi-variate correlation
brightness = (vrat/rat).mean(1)
templates  = np.vstack([o3.values,pm25.values,temps.values,humid.values,
                        np.ones_like(o3.values)]).T

ind = brightness<2.0
sol = np.linalg.lstsq(templates[ind],brightness[ind])

pred = np.dot(templates[ind],sol[0])



close("all")
dark_plot()
figure()
plot(o3[ind],brightness[ind],'.')
mo, bo = np.polyfit(o3[ind],brightness[ind],1)
R2 = 1-((brightness[ind]-(mo*o3[ind]+bo))**2).sum() / \
    ((brightness[ind]-brightness[ind].mean())**2).sum()
plot(o3,mo*o3+bo)
title(r"$R^{2}=%4.2f$" % R2)
ylabel("brightness")
xlabel("O3 [ppm]")
#savefig("../output/brightness_O3_veg.png", facecolor="k", clobber=True)

figure()
plot(pm25[ind],brightness[ind],'.')
mp, bp = np.polyfit(pm25[ind],brightness[ind],1)
R2 = 1-((brightness[ind]-(mp*pm25[ind]+bp))**2).sum() / \
    ((brightness[ind]-brightness[ind].mean())**2).sum()
plot(pm25,mp*pm25+bp)
title(r"$R^{2}=%4.2f$" % R2)
ylabel("brightness")
xlabel("PM2.5 [ppm]")
#savefig("../output/brightness_PM25_veg.png", facecolor="k", clobber=True)

figure()
plot(temps[ind],brightness[ind],'.')
mt, bt = np.polyfit(temps[ind],brightness[ind],1)
R2 = 1-((brightness[ind]-(mt*temps[ind]+bt))**2).sum() / \
    ((brightness[ind]-brightness[ind].mean())**2).sum()
plot(temps,mt*temps+bt)
title(r"$R^{2}=%4.2f$" % R2)
ylabel("brightness")
xlabel("T [F]")
#savefig("../output/brightness_T_veg.png", facecolor="k", clobber=True)

figure()
plot(humid[ind],brightness[ind],'.')
mh, bh = np.polyfit(humid[ind],brightness[ind],1)
R2 = 1-((brightness[ind]-(mh*humid[ind]+bh))**2).sum() / \
    ((brightness[ind]-brightness[ind].mean())**2).sum()
plot(humid,mh*humid+bh)
title(r"$R^{2}=%4.2f$" % R2)
ylabel("brightness")
xlabel("Humidity [%]")
#savefig("../output/brightness_H_veg.png", facecolor="k", clobber=True)

figure()
lin0, = plot(brightness[ind],lw=1)
lin1, = plot(pred)
xlabel("scan number")
ylabel("\"brightness\"")
legend([lin0,lin1],["data", "model"],loc="lower left")
title("O3, PM2.5, T, Humid regression")
#savefig("../output/brightness_model_veg.png", facecolor="k", clobber=True)

figure()
plot(o3*10000)
imshow((vrat/rat).T,clim=[0.6,2.0])
gca().set_yticks([np.argmin(np.abs(waves-i)) for i in range(400,1000,100)])
gca().set_yticklabels([str(i) for i in np.arange(0.4,1.0,0.1)])
gcf().canvas.draw()
title("red line scales as O3 ppm")
xlabel("scan number")
ylabel("wavelength")
#savefig("../output/scaled_spectra_veg.png", facecolor="k", clobber=True)




# -- check variation at specific values of humidity, temp, and ozone













# close("all")
# figure()
# plot(o3[ind],brightness[ind],'.')
# mo, bo = np.polyfit(o3[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mo*o3[ind]+bo))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(o3,mo*o3+bo)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("O3 [ppm]")

# figure()
# plot(pm25[ind],brightness[ind],'.')
# mp, bp = np.polyfit(pm25[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mp*pm25[ind]+bp))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(pm25,mp*pm25+bp)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("PM2.5 [ppm]")

# figure()
# plot(temps[ind],brightness[ind],'.')
# mt, bt = np.polyfit(temps[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mt*temps[ind]+bt))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(temps,mt*temps+bt)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("T [F]")

# figure()
# plot(humid[ind],brightness[ind],'.')
# mh, bh = np.polyfit(humid[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mh*humid[ind]+bh))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(humid,mh*humid+bh)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("Humidity [%]")

# figure()
# lin0, = plot(brightness[ind],lw=1)
# lin1, = plot(pred)
# xlabel("scan number")
# ylabel("\"brightness\"")
# legend([lin0,lin1],["data", "model"],loc="lower left")
# title("O3, PM2.5, T, Humid regression")

# A = 


# figure()
# plot(o3,(vrat/rat).sum(1))
# clf()
# plot(o3,(vrat/rat).sum(1),'o')
# figure()
# plot(humid,(vrat/rat).sum(1),'o')
# close("all")

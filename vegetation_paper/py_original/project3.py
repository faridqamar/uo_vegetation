import os
import time
import numpy as np
import hyss_util as hu
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.cluster import KMeans

# -- set data directories
DATADIR = os.path.join("..","data")
OUTDIR  = os.path.join("..","output")


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


# -- read in the data
try:
    cube
except:
    fname = "veg_00055.raw"
    hdr   = hu.read_header(os.path.join(DATADIR,fname.replace("raw","hdr")))
    t0    = time.time()
    cube  = hu.read_hyper(os.path.join(DATADIR,fname))
    dt    = time.time() - t0
    print("total read time is {0}m:{1:02}s".format(int(dt//60),int(dt%60)))


# -- read in the K-Means result
km = pkl.load(open("../output/km_00055_ncl10_seed314.pkl"))


# -- tag the K-Means of the full data set
try:
    dlabs = np.load("../output/km_00055_ncl10_see314_labs_full.npy")
except:
    print("normalizing full data cube...")
    data_norm = 1.0*cube.data.reshape(cube.nwav,cube.nrow*cube.ncol).T.copy()
    data_norm -= data_norm.mean(1,keepdims=True)
    data_norm /= data_norm.std(1,keepdims=True)

    print("predicting labels...")
    dlabs = km.predict(data_norm)

    np.save("../output/km_00055_ncl10_see314_labs_full.npy",dlabs)


# -- get just the vegetation spectra
sh       = cube.data.shape[1:3]
trind    = (dlabs==2)|(dlabs==5)
vspecs   = cube.data[:,trind.reshape(sh)]
veg_norm = (vspecs-vspecs.mean(0))/vspecs.std(0)


# -- get the sky spectrum and plot the reflectance
sspt = cube.data[:,:700].mean(-1).mean(-1)
plot(cube.waves,vspecs[:,::100]/sspt[:,np.newaxis],color='k',lw=0.01)


# -- cluster the total brightness of KM results from 10% of sample
np.random.seed(812)
clind = np.argsort(np.random.rand(int(0.1*vspecs.shape[1])))
n_clusters   = 10
n_jobs       = 16
random_state = 314
km_sub       = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                      random_state=random_state)
km_sub.fit(veg_norm[:,clind].T)


# -- cluster the reflectance of KM results
ref_norm  = vspecs/sspt[:,np.newaxis]
ref_norm -= ref_norm.mean(0)
ref_norm /= ref_norm.std(0)

n_clusters   = 10
n_jobs       = 16
random_state = 314
km_ref       = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                      random_state=random_state)
km_ref.fit(ref_norm[:,clind].T)
rlabs        = km_ref.predict(ref_norm.T)

# -- plot these clusters
clrs = plt.cm.Paired(np.linspace(0,1,n_clusters))[:,:3]
rags = np.zeros(list(sh) + [3], dtype=np.uint8)

rags[trind.reshape(sh)] = (255*clrs[rlabs]).astype(np.uint8)

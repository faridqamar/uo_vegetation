#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
good   = np.array([int(i) for i in np.load("../output/good_scans.npy")])
sc_sub = sc[sc.filename.isin(["veg_{0:05}.raw".format(i) for i in good])]

temps = sc_sub.temperature.values
humid = sc_sub.humidity.values
pm25  = sc_sub.pm25.values
o3   = sc_sub.o3.values


# -- get the time in seconds
secs = []
mndy = []
for stime in sc_sub.time.values:
    yr, mo, dy, tm = stime.split()
    stime_fmt      = "{0} {1} {2:02} {3}:00".format(yr,mo,int(dy),tm)
    obs_dt         = datetime.strptime(stime_fmt,"%Y %b %d %H:%M:%S")
    secs.append(float(obs_dt.strftime("%s")))
    mndy.append("{0} {1:02}".format(mo,int(dy)))
secs = np.array(secs)
mndy = np.array(mndy)


# -- plot the cadence
plt.close("all")

tstr = ["{0} {1:02}".format("May",i) for i in range(1,32)] + \
    ["{0} {1:02}".format("Jun",i) for i in range(1,8)]
tsec = [float(datetime.strptime("2016 "+i,"%Y %b %d").strftime("%s")) 
        for i in tstr]

clr = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
fig, ax = plt.subplots(figsize=[10,2])
fig.subplots_adjust(0.05,0.35,0.95,0.85)
fig.set_facecolor("w")
ax.set_axis_bgcolor("#EEEEEE")
ax.set_yticklabels("")
ax.set_xlim(tsec[0],tsec[-1])
for ii,sec in enumerate(secs):
    ax.plot([sec,sec],[0,1],color=clr[ii])
ax.set_xticks(tsec)
ax.set_xticklabels([i if i.split()[1]=="01" else i.split()[1] for i in tstr], 
                   rotation=90)
ax.xaxis.grid(True)
xr = ax.get_xlim()
yr = ax.get_ylim()
ax.text(xr[0],yr[1]+0.02*(yr[1]-yr[0]),"2016")
ax.text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),"{0} total scans".format(secs.size),
        ha="right")

cb = fig.add_axes((0.25,0.05,0.5,0.1))
cb.set_xticklabels("")
cb.set_yticklabels("")
cb.imshow(np.arange(100).reshape(2,50) % 50,"jet")
fig.text(0.245,0.05,"18%",ha="right",va="bottom")
fig.text(0.755,0.05,"99%",ha="left",va="bottom")
fig.text(0.5,0.05,"Humidity",ha="center",va="bottom")
fig.canvas.draw()
fig.savefig("../output/scan_times.pdf", clobber=True)
fig.savefig("../output/scan_times.png", clobber=True)
fig.savefig("../output/scan_times.eps", clobber=True)

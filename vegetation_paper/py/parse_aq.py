#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# -- utilities
AQDIR    = os.path.join("..","output","air_quality")
pm25_all = []
o3_all   = []
times    = []

# -- read in the data
for ii in range(38):
    if ii==0:
        fname = "StationData.csv"
    else:
        fname = "StationData ({0}).csv".format(ii)

    aq = pd.read_csv(os.path.join(AQDIR,fname),header=[0,1,2],nrows=24)

    # -- get only AQ measurements
    pm25 = aq[[i for i in aq.columns if "PM" in i[1]]]
    o3   = aq[[i for i in aq.columns if "O3" in i[1]]]

    # -- get times and mean AQ
    times.append(aq[aq.columns[1]])

    raw_pm25_vals = pm25.values
    raw_o3_vals   = o3.values

    pm25_vals = np.zeros(raw_pm25_vals.shape)
    o3_vals   = np.zeros(raw_o3_vals.shape)

    # -- convert missing values to NaN
    for ii in range(pm25_vals.shape[0]):
        for jj in range(pm25_vals.shape[1]):
            try:
                pm25_vals[ii,jj] = float(raw_pm25_vals[ii,jj])
            except:
                pm25_vals[ii,jj] = np.nan

    # -- convert missing values to NaN
    for ii in range(o3_vals.shape[0]):
        for jj in range(o3_vals.shape[1]):
            try:
                o3_vals[ii,jj] = float(raw_o3_vals[ii,jj])
            except:
                o3_vals[ii,jj] = np.nan

    # -- average over stations
    pm25_all.append(np.nanmean(pm25_vals,1))
    o3_all.append(np.nanmean(o3_vals,1))


# -- stack times and PM2.5 together write to csv
pm25_all = np.hstack(pm25_all)
o3_all   = np.hstack(o3_all)
times    = np.hstack(times)

fopen = open("../output/aq_vs_time.csv","w")
fopen.write("time,PM2_5,O3\n")

for ttime,tpm,to3 in zip(times,pm25_all,o3_all):
    fopen.write("{0},{1},{2}\n".format(" ".join(ttime.split()),tpm,to3))

fopen.close()

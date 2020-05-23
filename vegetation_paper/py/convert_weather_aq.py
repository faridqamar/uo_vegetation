#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# -- utilities
WUDIR = os.path.join("..","output","weather")
AQDIR = os.path.join("..","output")


# -- read in the times of the scans
scan_times = pd.read_csv("../output/scan_times.csv")


# -- initialize additional atributes
tempF = np.zeros(len(scan_times))
dewpt = np.zeros_like(tempF)
press = np.zeros_like(tempF)
humid = np.zeros_like(tempF)
preci = np.zeros_like(tempF)
pm25  = np.zeros_like(tempF)
o3    = np.zeros_like(tempF)


# -- for each scan time, get the date, and find the closest temperature
for ii,stime in enumerate(scan_times.time):

    # -- alert user
    if (ii+1)%50==0:
        print("\rworking on scan {0} of {1}...".format(ii+1,len(scan_times))),
        sys.stdout.flush()

    # -- parse the scan time
    yr, mo, dy, tm = stime.split()
    stime_fmt      = "{0} {1} {2:02} {3}:00".format(yr,mo,int(dy),tm)
    obs_dt         = datetime.strptime(stime_fmt,"%Y %b %d %H:%M:%S")

    # -- identify the WU file of the same date
    wufile = "wu_{0:02}{1:02}{2}.csv" \
        .format(obs_dt.month,obs_dt.day,obs_dt.year)

    # -- read WU file and initialize delta times
    wu  = pd.read_csv(os.path.join(WUDIR,wufile))
    dts = np.zeros(len(wu))

    # -- find the closest time
    for jj,ttime in enumerate(wu.Time):
        tdt     = datetime.strptime(ttime,"%Y-%m-%d %H:%M:%S")
        dts[jj] = (tdt-obs_dt).total_seconds()
    ind = np.argmin(np.abs(dts))

    # -- pull off weather parameters
    tempF[ii] = wu.TemperatureF[ind]
    dewpt[ii] = wu.DewpointF[ind]
    press[ii] = wu.PressureIn[ind]
    humid[ii] = wu.Humidity[ind]
    preci[ii] = wu.HourlyPrecipIn[ind]


# -- read in the air quality data
aq  = pd.read_csv(os.path.join(AQDIR,"aq_vs_time.csv"))


# -- for each scan time, get the date, and find the closest temperature
for ii,stime in enumerate(scan_times.time):

    # -- alert user
    if (ii+1)%50==0:
        print("\rworking on scan {0} of {1}...".format(ii+1,len(scan_times))),
        sys.stdout.flush()

    # -- parse the scan time
    yr, mo, dy, tm = stime.split()
    stime_fmt      = "{0} {1} {2:02} {3}:00".format(yr,mo,int(dy),tm)
    obs_dt         = datetime.strptime(stime_fmt,"%Y %b %d %H:%M:%S")

    # -- initialize delta times
    dts = np.zeros(len(aq))

    # -- find the closest time
    for jj,ttime in enumerate(aq.time):
        mo, dy, yrtime = ttime.split("/")
        yr, hrmn, pm   = yrtime.split(" ")
        hr, mn         = hrmn.split(":")
        aq_time = "{0:02}/{1:02}/{2} {3:02}:{4} {5}".format(int(mo),int(dy),yr,
                                                        int(hr),mn,pm)
        tdt     = datetime.strptime(ttime,"%m/%d/%Y %I:%M %p")
        dts[jj] = (tdt-obs_dt).total_seconds()
    ind = np.argmin(np.abs(dts))

    # -- pull off weather parameters
    pm25[ii] = aq.PM2_5[ind]
    o3[ii]   = aq.O3[ind]


# -- pack into a separate data frame
scan_cond = scan_times.copy()
scan_cond["temperature"]          = tempF
scan_cond["dew_point"]            = dewpt
scan_cond["pressure"]             = press
scan_cond["humidity"]             = humid
scan_cond["hourly_precipitation"] = preci
scan_cond["pm25"]                 = pm25
scan_cond["o3"]                   = o3


# -- write to file
scan_cond.to_csv("../output/scan_conditions.csv")

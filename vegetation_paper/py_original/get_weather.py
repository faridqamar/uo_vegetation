#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

# -- utilities
OUTDIR = os.path.join("..","output","weather")


# -- set the dates
dates = [[5,2,2016], [5,3,2016], [5,4,2016], [5,5,2016], [5,6,2016], 
         [5,7,2016], [5,8,2016], [5,9,2016], [5,10,2016], [5,11,2016], 
         [5,12,2016], [5,13,2016], [5,14,2016], [5,15,2016], [5,16,2016],
         [5,17,2016], [5,18,2016], [5,19,2016], [5,20,2016], [5,21,2016],
         [5,22,2016], [5,23,2016], [5,24,2016], [5,25,2016], [5,26,2016],
         [5,27,2016], [5,28,2016], [5,29,2016], [5,30,2016], [5,31,2016],
         [6,1,2016], [6,2,2016], [6,3,2016], [6,4,2016], [6,5,2016],
         [6,6,2016], [6,7,2016], [6,8,2016]]


# -- loop through and get data
for date in dates:

    # -- alert user
    print("\rgetting weather data for {0:02}/{1:02}/{2:4}...".format(*date)),
    sys.stdout.flush()

    # -- set url
    url = "https://www.wunderground.com/weatherstation/" + \
        "WXDailyHistory.asp?" + \
        "ID=KNYNEWYO116&" + \
        "day={0}&".format(date[1]) + \
        "month={0}&".format(date[0]) + \
        "year={0}&".format(date[2]) + \
        "graphspan=day&format=1"

    # -- read data from url
    cols = pd.read_csv(url, nrows=1).columns
    data = pd.read_csv(url, names=cols, header=False, skiprows=1, 
                       usecols=cols)

    # -- write to file
    outname = os.path.join(OUTDIR,"wu_{0:02}{1:02}{2:4}.csv".format(*date))
    data[::2].to_csv(outname)

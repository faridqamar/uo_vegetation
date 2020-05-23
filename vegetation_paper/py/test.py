import time
import numpy as np
import hyss_util as hu

# data = np.memmap("../data/veg_00000.raw",np.uint16,"r")

nrow = 1600
ncol = 1601
nwav = 848

# start = 21*nrow*nwav + (nrow-(5+1))
# end   = start + nrow*nwav
# every = nrow

data1 = np.memmap("../data/veg_00001.raw",np.uint16,mode="r") \
    .reshape(ncol,nwav,nrow)[:,:,::-1] \
    .transpose(1,2,0)

data2 = np.memmap("../data/veg_00002.raw",np.uint16,mode="r",
                  shape=(ncol,nwav,nrow))


t0 = time.time()
sky1 = data1[:,:700,:].mean(-1).mean(-1)
dt1 = time.time() - t0

t0 = time.time()
sky2 = data2[:,:,-700:].mean(-1).mean(0)
dt2 = time.time() - t0

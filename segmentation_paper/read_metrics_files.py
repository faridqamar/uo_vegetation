## this python code is to read the metrics in the metrics text files

import numpy as np
import matplotlib.pyplot as plt

# setup array with the total number of channels (each bin contains 848/#channels)
#channels = np.array([848, 424, 283, 212, 170, 142, 122, 106, 85, 53, 22])
channels = np.array([848, 424, 283, 212, 170, 142, 122])

testimage = ['108', '000', 'north']

accuracy = {'108':[], '000':[], 'north':[]}
avg_prec = {'108':[], '000':[], 'north':[]}
avg_recall = {'108':[], '000':[], 'north':[]}
veg_prec = {'108':[], '000':[], 'north':[]}
veg_recall = {'108':[], '000':[], 'north':[]}

# loop through files reading:
# accuracy, avg precision and recall, and vegetation precision and recall

for testim in testimage:
    for channel in channels:
        filename = "./metrics/CNN_spatial_train_108_test_" + testim + \
                    "_binned_" + str(channel) + "_metrics.txt"
        f = open(filename, "r")
        f1 = f.readlines()

        # read accuracy from 13th line
        accuracy[testim].append(float(f1[12]))

        # read avg_prec and avg_recall from 28th line
        Nums = []
        for num in f1[27].split():
            try:
                Nums.append(float(num))
            except ValueError:
                pass
        avg_prec[testim].append(Nums[0])
        avg_recall[testim].append(Nums[1])        
        
        # read veg_prec and veg_recall from 20th line
        Nums = []
        for num in f1[19].split():
            try:
                Nums.append(float(num))
            except ValueError:
                pass
        veg_prec[testim].append(Nums[0])
        veg_recall[testim].append(Nums[1])  
        f.close()

print(accuracy)
print(avg_prec)
print(avg_recall)

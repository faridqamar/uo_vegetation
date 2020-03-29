import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
       'size' : 20}
matplotlib.rc('font', **font)
 
labels = ['sky', 'clouds','vegetation', 'water', 'built',
          'windows', 'roads', 'cars', 'metal']

colors = [[0,0.32549,0.62353,1], [0.93333,0.9098,0.77255,1], [0,0.61961,0.45098,1],  [0.33725,0.70588,0.91373,1],
        [0,0,0,1], [1,0.82353,0,1], [0.90196,0.62353,0,1], [0.83529,0.36863,0,1], [0.8,0.47451,0.65490,1]]

fig = plt.figure(figsize=(20, 2))
patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(labels, colors)]
fig.legend(patches, labels, loc='center', frameon=True, mode="expand", ncol=18)
plt.show()
fig.savefig("./segmentation_paper/plots/Legend.png", bbox_inches='tight')
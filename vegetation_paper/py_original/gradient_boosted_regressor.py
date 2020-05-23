#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import GradientBoostingRegressor

rind = np.argsort(np.random.rand(1102))
trind = rind[:770]
teind = rind[770:]

train = np.vstack([brat[trind].T,o3[trind],pm25[trind],temps[trind],
                   humid[trind]]).T
test = np.vstack([brat[teind].T,o3[teind],pm25[teind],temps[teind],
                  humid[teind]]).T

gbr3 = GradientBoostingRegressor(n_estimators=500,max_depth=6)
gbr3.fit(train,ndvi[trind])
predtrain = gbr3.predict(train)
pred3 = gbr3.predict(test)

close("all")
linte, = plot(ndvi[teind],pred3,'.',color="darkred")
lintr, = plot(ndvi[trind],predtrain,'.',color="darkorange")
linlin, = plot(ndvi[teind],ndvi[teind],color="dodgerblue",lw=0.5)
xlabel("NDVI measured")
ylabel("NDVI predicted")
legend([lintr,linte],["training set","testing set"],loc="upper left")
savefig("../output/predict_ndvi.pdf")



plt.close("all")
figure(figsize=(6.5,4))
subplots_adjust(0.15,0.15,0.95,0.9)
impwgt = gbr3.feature_importances_[:848]
fill_between(waves*1e-3,0,impwgt,facecolor="dodgerblue",edgecolor="darkblue")
xlim(waves[0]*1e-3,waves[-1]*1e-3)
xlabel("wavelength [micron]")
ylabel("importance [arb units]")
xr, yr = gca().get_xlim(), gca().get_ylim()
gca().text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),"Feature importances",fontsize=14,
         ha="right")
gcf().canvas.draw()
savefig("../output/predict_ndvi_fi.pdf", clobber=True)

## this python code is to open an image and select pixels to add to a file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

val = 0
classType = ['sky',
			'clouds',
			'vegetation',
			'water',
			'buildings',
			'windows',
			'roads',
			'cars',
			'metal']
numLines = 0

def getClassType(val, classType):
	classType = ['sky',
				'clouds',
				'vegetation',
				'water',
				'buildings',
				'windows',
				'roads',
				'cars',
				'metal']
	print("Which class of pixels are you selecting?")
	print("Enter the number associate with the class you wish to select:")
	print("1) %s" %(classType[0]))
	print("2) %s" %(classType[1]))
	print("3) %s" %(classType[2]))
	print("4) %s" %(classType[3]))
	print("5) %s" %(classType[4]))
	print("6) %s" %(classType[5]))
	print("7) %s" %(classType[6]))
	print("8) %s" %(classType[7]))
	print("9) %s" %(classType[8]))
	class_type = int(input("class type = ")) - 1
	return class_type


while(1):
	class_type = getClassType(val, classType)	
	try:
		val = int(class_type)
		if(val in [0, 1, 2, 3, 4, 5, 6, 7, 8]):
			print("%s class type selected" %(classType[val].upper()))
			break
		else:
			print("")			
			print("ERROR: The class type selected is invalid! Try again")
	except ValueError:
		print("")
		print("ERROR: The class type selected is invalid! Try again")
			 
filename = "./" + str(val+1) +"_" + classType[class_type] + "_coordinates_108_for_distance.txt"
#filename = "./manual_classified_pixels/" + str(val+1) +"_" + classType[class_type] + "_coordinates_000.txt"
#filename = "./manual_classified_pixels/" + str(val+1) +"_" + classType[class_type] + "_coordinates_north.txt"
try:
	f = open(filename, "r")
	f1 = f.readlines()
	numLines = len(f1)
	print("---")
	print("There are currently %d coordinates already in file %s" %(numLines, filename))
	f.close()
	print("---")
except FileNotFoundError:
	print("---")


img = mpimg.imread('./output/scene_RGB_00108.png')
#img = mpimg.imread('./output/scene_RGB_00000.png')
#img = mpimg.imread('./output/scene_RGB_gimp_corrected_north.png')

xpixels, ypixels = 1600, 1600	# for south facing images
#xpixels, ypixels = 1247, 1600	# for north facing images
fig = plt.figure(figsize=(20, 10), dpi=80)
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax.imshow(img, interpolation='none', aspect=0.5)

def onclick(event):
    if event.dblclick:
        circle = plt.Circle((event.xdata, event.ydata), 1, color='blue')
        ax.add_patch(circle)
        fig.canvas.draw()

        cind = int(round(event.xdata))
        rind = int(round(event.ydata))
        global numLines
        numLines += 1
        print("%d: (%d, %d)" % (numLines, rind, cind))

        f = open(filename, "a+")
        f.write("%d %d\n" % (rind, cind))
        f.close()
    
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()



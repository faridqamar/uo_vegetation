## this python code is to open an image and select pixels to add to a file

import numpy as np
import cv2

val = 0
classType = ['sky', 'vegetation', 'built']

def getClassType(val, classType):
	classType = ['sky', 'vegetation', 'built']
	print("Which class of pixels are you selecting?")
	print("Enter the number associate with the class you wish to select:")
	print("1) %s" %(classType[0]))
	print("2) %s" %(classType[1]))
	print("3) %s" %(classType[2]))
	class_type = int(input("class type = ")) - 1
	return class_type

while(1):
	class_type = getClassType(val, classType)	
	try:
		val = int(class_type)
		if(val in [0, 1, 2]):
			print("%s class type selected" %(classType[val].upper()))
			break
		else:
			print("")			
			print("ERROR: The class type selected is invalid! Try again")
	except ValueError:
		print("")
		print("ERROR: The class type selected is invalid! Try again")
			 
filename = "./" + classType[class_type] + "_coordinates.txt"
try:
	f = open(filename, "r")
	f1 = f.readlines()
	print("---")
	print("There are currently %d coordinates already in file %s" %(len(f1), filename))
	f.close()
	print("---")
except FileNotFoundError:
	print("---")

xcoord = []
ycoord = []

def draw_circle(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
		xcoord.append(x)
		ycoord.append(y)
		print("(%d, %d)" %(x, y))

# read image
img = cv2.imread('./output/scene_RGB_00108.png')
cv2.namedWindow('./output/scene_RGB_00108.png')
cv2.setMouseCallback('./output/scene_RGB_00108.png', draw_circle)

# open image and begin selecting pixels
while(1):
	cv2.imshow('./output/scene_RGB_00108.png', img)
	k = cv2.waitKey(20) & 0xFF
	# if 10 pixels are selected, print them to file
	if len(xcoord) == 10:
		print("10 coordinates selected")
		print("Printing coordinate list to file %s" %filename)
		f = open(filename, "a+")
		for i in range(len(xcoord)):
			f.write("%d %d\n" %(xcoord[i], ycoord[i]))
		f.close()
		print("%d coordinates successfully written to file %s" %(len(xcoord), filename))
		xcoord = []
		ycoord = []
	# to exit click the Esc button, any selected pixels will be added to the file
	if k == 27:
		print("Printing coordinate list to file %s" %filename)
		f = open(filename, "a+")
		for i in range(len(xcoord)):
			f.write("%d %d\n" %(xcoord[i], ycoord[i]))
		f.close()
		print("%d coordinates successfully written to file %s" %(len(xcoord), filename))
		break
	#elif k == ord('a'):
	#	print(mouseX, mouseY)






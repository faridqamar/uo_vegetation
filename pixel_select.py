## this python code is to open an image and select pixels to add to a file

import numpy as np
import cv2
import json

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
			print("ERROR: The class type selected is invalid please try again")
	except ValueError:
		print("")
		print("ERROR: The class type selected is invalid please try again")
			 
filename = "./" + classType[class_type] + "_coordinates.txt"
xcoord = []
ycoord = []

def draw_circle(event, x, y, flags, param):
	#global mouseX, mouseY
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
		#mouseX, mouseY = x, y
		xcoord.append(x)
		ycoord.append(y)
		print("(%d, %d)" %(x, y))

img = cv2.imread('./output/scene_RGB_00108.png')
cv2.namedWindow('./output/scene_RGB_00108.png')
cv2.setMouseCallback('./output/scene_RGB_00108.png', draw_circle)

while(1):
	cv2.imshow('./output/scene_RGB_00108.png', img)
	k = cv2.waitKey(20) & 0xFF
	if k == 27:
		print("Printing coordinate list to file %s" %filename)
		f = open(filename, "a+")
		for i in range(len(xcoord)):
			f.write("%d %d\n" %(xcoord[i], ycoord[i]))
		f.close()
		break
	#elif k == ord('a'):
	#	print(mouseX, mouseY)






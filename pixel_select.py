## this python code is to open an image and select pixels to add to a file

import numpy as np
import cv2

coordList = dict(x=[], y=[])

def draw_circle(event, x, y, flags, param):
	#global mouseX, mouseY
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
		#mouseX, mouseY = x, y
		coordList['x'].append(x)
		coordList['y'].append(y)
		print("(%d, %d)" %(x, y))

img = cv2.imread('./output/scene_RGB_00108.png')
cv2.namedWindow('./output/scene_RGB_00108.png')
cv2.setMouseCallback('./output/scene_RGB_00108.png', draw_circle)

while(1):
	cv2.imshow('./output/scene_RGB_00108.png', img)
	k = cv2.waitKey(20) & 0xFF
	#print(mouseX, mouseY)
	if k == 27:
		print("Printing coordinate list to file")# %(len(coordList['x'])))
		f = open("./coordinates.txt", "w")
		f.write(str(coordList))
		f.close()
		break
	#elif k == ord('a'):
	#	print(mouseX, mouseY)

## this python code is to open an image and select pixels to add to a file

import numpy as np
import cv2

class LoadImage:
	def loadImage(self):
		self.img-=cv2.imread('./output/scene_RGB_00108.png')
		cv2.imshow('Test', self.img)
		
		self.pressedkey = cv2.waitKey(0)

		#Wait for Esc key to exit
		if self.pressedkey==27:
			cv2.destroyAllWindows()

	#Start of the main program here
	if __name__ = "__main__":
		LI = LoadImage()
		LI..loadImage()

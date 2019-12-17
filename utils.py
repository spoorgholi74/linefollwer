import numpy as np
import cv2
import imutils
import math
import glob
from matplotlib import pyplot as plt
 
def contrast(image, a, b):
	alpha = 2 # Simple contrast control[1-3]
	beta = 0    # Simple brightness control [0-100]
	image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	#cv2.imshow('Contrasted', image)
	#cv2.waitKey(0)
	return image


def resize(image, w_size):
	ratio = float(w_size) / image.shape[1]
	dim = (w_size, int(image.shape[0] * ratio))
	resizedCubic = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
	return resizedCubic


def adaptiveThreshold(image, method):
	gray = contrast(image, a = 1, b = 0)

	if method == 'binary':
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
		#cv2.imshow('binary thresh', thresh)
	elif method == 'otsu':
		blur = cv2.GaussianBlur(gray,(5,5),0)
		#blur = cv2.bilateralFilter(gray, 11, 17, 17)
		high_thresh,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#cv2.imshow('OTSU thresh', thresh)
	elif method == 'adaptive':
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
		#thresh1 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		#cv2.imshow('adaptive guassian thresh', thresh)
	else:
		raise Exception("The method is not valid!")

	#cv2.imwrite("processed//plate{}.jpg".format(i), thresh)
	return thresh

def detect(c, ratio):
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	(x, y, w, h) = cv2.boundingRect(approx)
	c *= int(ratio)
	M = cv2.moments(c)
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	if M["m00"] != 0:
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)
	else:
		cX = 0
		cY = 0
	#cv2.circle(image, (cX, cY - int((h/2)*ratio)), 7, (0, 255, 0), -1)
	#cv2.circle(image, (cX, cY + int((h/2)*ratio)), 7, (0, 0, 255), -1)

	ratio = int(ratio)
	extLeft = tuple(c[c[:, :, 0].argmin()][0] * ratio) 
	extRight = tuple(c[c[:, :, 0].argmax()][0] * ratio)
	extTop = tuple(c[c[:, :, 1].argmin()][0] * ratio)
	extBot = tuple(c[c[:, :, 1].argmax()][0] * ratio)
	'''
	cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(image, extRight, 8, (0, 255, 0), -1)
	cv2.circle(image, extTop, 8, (255, 0, 0), -1)
	cv2.circle(image, extBot, 8, (255, 255, 0), -1)
	'''
	#cv2.line(image, extBot, extTop, (0, 0, 255), 3)
	ar = w / float(h)
	if len(approx) == 4:
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
		#print(shape)
	else:
		shape = "None"
		#print("shape is not a rectangle")
	return cX, cY, shape


def filter_color(img):
	# converting from BGR to HSV color space
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	 
	# Range for lower red
	lower_black = np.array([0,0,0])
	upper_black = np.array([20,20,60])
	mask1 = cv2.inRange(hsv, lower_black, upper_black)
	se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 8))
	morph_img = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, se)
	#cv2.imshow('morph_mask', morph_img)
	#cv2.waitKey(0)
	return morph_img


# Read image
def process(image):
	height, width = image.shape[:2]
	image_center = [int(width/2), int(height/2)]
	resized = imutils.resize(image, width=width)
	ratio = image.shape[0] / float(resized.shape[0])
	mask = filter_color(resized) 
	masked = cv2.bitwise_and(resized, resized, mask = mask)
	#cv2.imshow("after mask", masked)
	gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	thresh = adaptiveThreshold(blurred, 'otsu')
	cv2.namedWindow('thresh1', cv2.WINDOW_NORMAL)
	cv2.imshow("thresh1", thresh)
	cv2.resizeWindow('thresh1', 600,600)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#find the biggest area
	cnts = max(cnts, key = cv2.contourArea)
	#np array

	cv2.circle(image, (image_center[0], image_center[1]), 7, (255, 0, 255), -1)
	cv2.putText(image, "I", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)
	#print("The cnts are :", cnts)
	# loop over the contours
	'''
	kpCnt = len(cnts[0])
	print(kpCnt)
	x = 0
	y = 0
	for kp in cnts[0]:
		print(kp)
		x = x+kp[0][0]
		y = y+kp[0][1]
	print(x/kpCnt, y/kpCnt)
	cv2.circle(image, (int(x/kpCnt), int(y/kpCnt)), 10, (0, 0, 255), -1)
	'''
	#for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	cX, cY, shape = detect(cnts, ratio)
	c = cnts.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

		

	distance = math.sqrt( ((image_center[0] - cX)**2)+((image_center[1] - cY)**2))
	print (distance, '\n')
	x_dif = image_center[0] - cX
	y_dif = image_center[1] - cY
	#print(x_dif, y_dif)

	cv2.line(image, (image_center[0], image_center[1]), (cX, cY), (0, 0, 255), 2)
	cv2.circle(image, (cX, cY), 7, (255, 0, 255), -1)
	cv2.putText(image, "L", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1, (255, 0, 0), 2)

	cv2.putText(image, "x_diff = ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)
	cv2.putText(image, str(x_dif), (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)

	cv2.putText(image, "y_diff = ", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)
	cv2.putText(image, str(y_dif), (200, 100), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)

	cv2.putText(image, "distance = ", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)
	cv2.putText(image, str(int(distance)), (200, 150), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 255, 0), 2)

	

	if x_dif < -120:
		command = "right"
	elif x_dif > 120:
		command = "left"
	else:
		command = "straight"

	print(command)
	cv2.putText(image, "Move = ", (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 0, 255), 2)
	cv2.putText(image, command, (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 0, 255), 2)


	cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
	cv2.imshow("Final", image)
	cv2.resizeWindow('Final', 600,600)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return image, command

def main():
	for plate in glob.glob("Data/*.jpg"):
		image = cv2.imread(plate)
		image, command = process(image = image )
		#filter_color(image)

if __name__ == '__main__':
	main()
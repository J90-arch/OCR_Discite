import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
def rotate(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle > 45:
	    angle = 90 - angle
	else:
	    angle = -angle
	if angle ==0:
		return (image, angle)
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	print("angle: {:.3f}".format(angle))
	#cv2.imshow("Input", image)
	#cv2.imshow("Rotated", rotated)
	#cv2.waitKey(0)
	return (rotated, angle)

#print(rotate(image))
x = rotate(image)
image = x[0]
angle = x[1]
while  angle != 0:
	x = rotate(image)
	image = x[0]
	angle = x[1]
cv2.imwrite(r'rotated.png', image)
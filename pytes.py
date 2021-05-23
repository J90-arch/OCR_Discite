import numpy as np
import argparse
import cv2
import pytesseract
import re
#from line_profiler import LineProfiler


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image file")
ap.add_argument("-f", "--file", required=True, help="file to write text to")
ap.add_argument("-l", "--lang", required=True, help="language to read")
args = vars(ap.parse_args())

def write_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

#@profile
def rotate(rotated):
	angle = 0.01
	while angle != 0:
		print("angle: {:.3f}".format(angle))
		gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
		gray = cv2.bitwise_not(gray)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		coords = np.column_stack(np.where(thresh > 0))
		angle = cv2.minAreaRect(coords)[-1]
		print("angle: {:.3f}".format(angle))
		if angle > 45:
		    angle = 90 - angle
		else:
		    angle = -angle
		#if angle ==0:
		#	break
		(h, w) = rotated.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(rotated, M, (w, h),
			flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	#cv2.imshow('rotated', rotated)
	#cv2.waitKey()
	return rotated

#@profile
def image_to_string(lan, img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.threshold(img, 128, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(img, lang=lan)
    return text
#@profile
def fix_text(text):
	text = re.sub("\n\s+", "\n", text)
	text = re.sub("-\n", "", text)
	text = re.sub("\|", "", text)
	return text

#@profile
def main():
	image = rotate(cv2.imread(args["image"]))
	cv2.imshow('image used for ocr', image)
	cv2.waitKey()
	text = image_to_string(args["lang"], image)
	write_file(fix_text(text), args["file"])


if __name__ == "__main__":
	main()



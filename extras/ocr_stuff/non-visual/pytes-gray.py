from PIL import Image

import pytesseract
import sys
import re
import cv2
image = cv2.imread('test.png')
#cv2.imshow('original', image)
#cv2.waitKey(0)
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh, black_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#cv2.imshow('black', black_image)
#cv2.waitKey(0)

text = pytesseract.image_to_string(black_image, lang='lit')
print(text)
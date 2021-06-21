#! /usr/bin/python
import cv2
import pytesseract
import numpy as np
from PIL import Image
import easyocr
language = 'lt'
img = 'test.png'
#def write_image(image, path):
#        img = Image.fromarray(np.array(image), 'L')
#        img.save(path)
#
#img = cv2.imread('C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\test.png')
#
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
#
#gray, img_bin = cv2.threshold(gray,127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#gray = cv2.bitwise_not(img_bin)
#
#write_image(gray, 'C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\example2.png')
#kernel = np.ones((2, 1), np.uint8)
#img = cv2.erode(gray, kernel, iterations=1)
#write_image(img, 'C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\example3.png')

#img = cv2.dilate(img, kernel, iterations=1)

#write_image(img, 'C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\example4.png')
reader = easyocr.Reader([language]) # need to run only once to load model into memory
result = reader.readtext(img)

#out_below = pytesseract.image_to_string(img)
#print("OUTPUT:", result)
#for x in result:
#	print(x[1])
color = (255, 0, 0)
photo = cv2.imread('C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\test.png')
for x in result:
    photo = cv2.rectangle(photo, tuple([int(y) for y in x[0][0]]), tuple([int(y) for y in x[0][2]]), color, 1)
#cv2.imshow('none', photo)
cv2.imwrite('C:\\Users\\jokub\\Documents\\random code\\pytesseract_stuff\\testb.png',photo)


print(reader.readtext(img, detail=0, paragraph=True))

import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray,175, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)
cv2.imwrite('gray.png', gray)

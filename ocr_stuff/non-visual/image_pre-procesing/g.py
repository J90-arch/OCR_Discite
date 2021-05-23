import cv2
import numpy as np
img = cv2.imread('0.jpg')
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('1resize.png', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('2gray.png', img)
cv2.imshow('GRAY',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite('3dilate.png', img)
cv2.imshow('DILATE',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.erode(img, kernel, iterations=1)
cv2.imwrite('4erode.png', img)
cv2.imshow('ERODE',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('5blur.png', img)
cv2.imshow('BLUR',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.threshold(img, 128, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imwrite('6thereshold.png', img)
cv2.imshow('THRESHOLD',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
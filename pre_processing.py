import cv2
import numpy as np
PATH = r"C:\Users\jokub\Desktop\Work\git_rep_ocr\extras\tests\IMG_0771.jpg"

img = cv2.imread(PATH)

'''
imgf = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image
cv2.imshow("Adaptive Thresholding",imgf)

#ret, imgf = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
#cv2.imshow("Otsu's binarization",imgf)

cv2.waitKey()
cv2.imwrite('img.png', imgf)
'''
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.threshold(img, 128, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cv2.imshow("current",img)

cv2.waitKey()
cv2.imwrite('img.png', img)
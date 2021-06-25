from PIL import Image
import numpy
from numpy.core.numeric import argwhere
import cv2
PATH = "test.jpg"
img = cv2.imread(PATH, 0)

img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = numpy.ones((1, 1), numpy.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
img = cv2.GaussianBlur(img, (5, 5), 0)
#img = cv2.threshold(img, 64, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7) #imgf contains Binary image
#img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7) 
#cv2.imshow("img", img)
#cv2.waitKey()
#print(img.shape)
cv2.imwrite("test-mono.jpg", img)

edges = numpy.zeros(img.shape, dtype = int)


#def find_first(x):
#    idx = x.view(bool).argmax() // x.itemsize
#    return idx if x[idx] else -1


#black_pixels = [[find_first(img[i]), find_first(img[i][::-1])] for i in range(len(img))]
#for item in black_pixels:
#    print(item)

for i in range(len(img[0])):
    black_pixel_before = True
    for j in range(len(img)):
        if img[-j][i] == 0 and black_pixel_before:
            edges[-j][i] = 255
            black_pixel_before = False
        elif img[-j][i] == 255 and not black_pixel_before:
            black_pixel_before = True


cv2.imwrite('img.png', edges)

#for item in img:
#    black_pixels = numpy.where(img == 255)
#    print(black_pixels)
#pixels = numpy.asarray(Image.open(img))
#print(pixels[0])
#img = Image.fromarray(numpy.uint8(pixels))

#cv2.imshow("current",img)

#cv2.waitKey()
#cv2.imwrite('img.png', img)
#cv2.imwrite('img.png', numpy.array([[256, 256], [256, 0]]))
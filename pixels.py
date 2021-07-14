from PIL import Image
import numpy as np
#from numpy.core.numeric import argwhere
import cv2
PATH = "test.jpg"
img = cv2.imread(PATH, 0)

img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
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

img = np.array(img, int)



img = np.rot90(img)#rotate image 90 degrees so columns are one list each
edges = np.zeros(img.shape, dtype = int)
#def find_first(x):
#    idx = x.view(bool).argmax() // x.itemsize
#    return idx if x[idx] else -1


#black_pixels = [[find_first(img[i]), find_first(img[i][::-1])] for i in range(len(img))]
#for item in black_pixels:
#    print(item)
'''
for i in range(len(img[0])):
    black_pixel_before = True
    #j for line, i for column
    for j in range(len(img)):
        if img[-j][i] == 0 and black_pixel_before:
            edges[-j][i] = 255
            black_pixel_before = False
        elif img[-j][i] == 255 and not black_pixel_before:
            black_pixel_before = True
'''
# we need to use negative j, because we want to evaluate pixels from bottom to the top
l = img.shape[1]//15
print(l)
for i in range(len(img)):
    for j in range(l -1, len(img[0])):
        if img[i][-j] == 0 and 0 not in img[i][-j+1:-j+l]:
            edges[i][-j] = 255
            j+=l

        
edges = np.rot90(edges, 3)
print(edges.shape)
#remove single white pixels

for i in range(1, edges.shape[0]-1):
    for j in range(1, edges.shape[1] -1):
        if edges[i][j] == 255:
            if edges[i][j-1] == 0 and edges[i][j+1] == 0 :
                edges[i][j] = 0




cv2.imwrite('img.png', edges)



#print(img.shape)
#img = np.rot90(img)
#print(img.shape)


#for item in img:
#    black_pixels = np.where(img == 255)
#    print(black_pixels)
#pixels = np.asarray(Image.open(img))
#print(pixels[0])
#img = Image.fromarray(np.uint8(pixels))

#cv2.imshow("current",img)

#cv2.waitKey()
#cv2.imwrite('img.png', img)
#cv2.imwrite('img.png', np.array([[256, 256], [256, 0]]))
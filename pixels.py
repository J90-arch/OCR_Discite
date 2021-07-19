from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
PATH = "test.jpg"
img = cv2.imread(PATH, 0)
###Code currently is held together by ducktape, so expect it to break
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

photo = False
if photo:
    edges = np.zeros(img.shape, dtype = int)
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

else:
    print(img.shape)
    base_pixels = [[],[]]
    for i in range(img.shape[0]-1):
        for j in range(l -1, img.shape[1]):
            if img[i][-j] == 0 and 0 not in img[i][-j+1:-j+l]:
                                
                base_pixels[0].append(i)#x
                base_pixels[1].append(j)#y

                j=0
                i+=1
    
    for i in range(1, len(base_pixels)-1):
        if base_pixels[0][i] != base_pixels[0][i-1] +1 and base_pixels[0][i] != base_pixels[0][i+1] -1:
            base_pixels[0].pop(i)
            base_pixels[1].pop(i)
    
    base_pixels[0].reverse()
    base_pixels[0] = [img.shape[0]-x for x in base_pixels[0]]
    base_pixels[1].reverse()
    plt.plot(base_pixels[0], base_pixels[1], "r+")
    plt.show()

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
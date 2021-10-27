import cv2
import numpy as np 
import math
import matplotlib.pyplot as plt
 
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7)

rows, cols = img.shape 
 

 
##################### 
# Horizontal wave 
 
img_output = np.zeros(img.shape, dtype=img.dtype) 
 
for i in range(rows): 
    for j in range(cols): 
        offset_x = 0 
        offset_y = int(16.0 * j / 150)
        if i+offset_y < rows: 
            img_output[i,j] = img[(i+offset_y)%rows,j] 
        else: 
            img_output[i,j] = 0
print(img_output)
plt.plot(img_output)
plt.show()
cv2.imshow('Horizontal wave', img_output)
cv2.waitKey() 
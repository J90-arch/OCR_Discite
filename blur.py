import cv2 
from matplotlib import pyplot as plt 
# Reading image from folder where it is stored 
img = cv2.imread('img.png') 
# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 
plt.show()
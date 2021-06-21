import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#rotate image
def rotate(image_path, degrees_to_rotate, saved_location):
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)

#get f transformation
def transform_data(m):
    dpix, dpiy = m.shape
    x_c, y_c = np.unravel_index(np.argmax(m), m.shape)
    angles = np.linspace(0, np.pi*2, min(dpix, dpiy))
    mrc = min(abs(x_c - dpix), abs(y_c - dpiy), x_c, y_c)
    radiuses = np.linspace(0, mrc, max(dpix, dpiy))
    A, R = np.meshgrid(angles, radiuses)
    X = R * np.cos(A)
    Y = R * np.sin(A)
    return A, R, m[X.astype(int) + mrc - 1, Y.astype(int) + mrc - 1]

img_c1 = cv2.imread("gray.png", 0)
img_c3 = np.fft.fftshift(np.fft.fft2(img_c1))
f = np.log(1+np.abs(img_c3))
angles, radiuses, m = transform_data(f)
n = []
for i in range(len(angles[0])):
        n.append(0.0)
        for j in range(len(angles)):
            n[i] += m[j][i]

a = angles[0][n.index(max(n[int((len(n)/8*3)):int((len(n)/8*5))]))]*180/np.pi
rotate("test.png", 180-a, "rgray.png")


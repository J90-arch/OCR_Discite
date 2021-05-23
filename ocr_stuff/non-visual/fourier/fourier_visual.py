import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def rotate(image_path, degrees_to_rotate, saved_location):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)
    rotated_image.show()

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
plt.imshow(img_c1, "gray"), plt.title("original")
#plt.show()

img_c2 = np.fft.fft2(img_c1)
plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("spectrum")
#plt.show()
img_c3 = np.fft.fftshift(img_c2)
f = np.log(1+np.abs(img_c3))
plt.imshow(f, "gray"), plt.title("Centered Spectrum")
plt.show()
#print(f)
angles, radiuses, m = transform_data(f)

plt.contourf(angles, radiuses, m)

plt.show()
print(f'angles {angles}')
print(f'radiuses{radiuses}')
print(f'm{m}')
print(angles[0])
print(len(angles))
print(len(angles[0]))
print(len(m))
print(len(m[0]))
n = []
for i in range(len(angles[0])):
        n.append(0.0)
        for j in range(len(angles)):
            n[i] += m[j][i]
print(len(n))
plt.plot([ 180*x/np.pi for x in angles[0]][int((len(n)/8*3)):int((len(n)/8*5))], n[int((len(n)/8*3)):int((len(n)/8*5))])
plt.show()
#sample_angles = np.linspace(0,  2 * np.pi, len(c.sum(axis=0))) / np.pi*180
#turn_angle_in_degrees = 90 - sample_angles[np.argmax(n.sum(axis=0))]
#plt.plot(sample_angles, c.sum(axis=0))
#plt.show()
#print(turn_angle_in_degrees)
print(len(n))
print(max(n[int((len(n)/8*3)):int((len(n)/8*5))]))
print(n.index(max(n[int((len(n)/8*3)):int((len(n)/8*5))])))
print(angles[0][n.index(max(n[int((len(n)/8*3)):int((len(n)/8*5))]))]*180/np.pi)
a = angles[0][n.index(max(n[int((len(n)/8*3)):int((len(n)/8*5))]))]*180/np.pi
rotate("test.png", 180-a, "rgray.png")


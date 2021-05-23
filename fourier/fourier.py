import cv2
import numpy as np
from matplotlib import pyplot as plt
PATH = r'C:\Users\jokub\Desktop\Work\fourier\0.png'
img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()


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

angles, radiuses, m = transform_data(magnitude_spectrum)

plt.contourf(angles, radiuses, m)
plt.show()

#print('f\n', f)
#print('fshift\n', fshift)
#print('spectrum\n', magnitude_spectrum)
#print('m\n', m)
#print('radiuses\n', radiuses)
#print('angles\n', angles)

c = []
print(np.shape(m))
for i in range(len(m[0])):
    a = [x[i] for x in m]
    c.append(a)
print('c\n', c[0][:10])
print(np.shape(c))
c = np.array(c)
sample_angles = np.linspace(0,  2 * np.pi, len(c.sum(axis=0))) / np.pi*180
turn_angle_in_degrees = 90 - sample_angles[np.argmax(c.sum(axis=0)[::-1])]
print(turn_angle_in_degrees %90)

plt.plot(sample_angles, c.sum(axis=0))
plt.show()
'''
print(c.sum(axis=0)[:20])
print(len(c.sum(axis=0)))

c_smooth = []
k = 2
for i in range(len(c.sum(axis=0))-k):
    a = 0
    for j in range(k+1):
        a += c.sum(axis=0)[i+j]
    c_smooth.append(a)

plt.plot(sample_angles[:-k], c_smooth)
plt.show()
'''
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
###Code currently is held together by ducktape, so expect it to break
def baseline_img(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.threshold(img, 64, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7) #imgf contains Binary image
    #img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7) 
    cv2.imwrite("test-mono.jpg", img)
    img = np.array(img, int)
    img = np.rot90(img)#rotate image 90 degrees so columns are one list each
    # we need to use negative j, because we want to evaluate pixels from bottom to the top
    l = img.shape[1]//15
    print(l)
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
    cv2.imwrite('tmp.png', edges)
    return edges

# Find a horizontal line with most pixels in middle half of the image
def find_whitest_line(img):
    number_of_white_pixels = 0
    for i in range(len(img)//4,len(img)//4*3):
        num = np.sum(img[i] == 255)
        if num >= number_of_white_pixels:
            number_of_white_pixels = num
            line_with_most_white_pixels = i
    return line_with_most_white_pixels

# Find first and last pixel in that line within 1/20 of image length to other white pixels
def points(whitest_line):
    for i in range(len(whitest_line)//20 , len(whitest_line)):
        if whitest_line[i] == 255 and 255 in whitest_line[i:i+len(whitest_line)//20]:
            First_white_pixel = i
            break
    for i in range(First_white_pixel , len(whitest_line)):
        if 255 not in whitest_line[i:i+len(whitest_line)//20]:
            Last_white_pixel = i-1
            break
    Approx_centre = int((First_white_pixel + Last_white_pixel)/2)
    return First_white_pixel,Last_white_pixel,Approx_centre

def main():
    PATH = "test.jpg"
    img = cv2.imread(PATH, 0)
    img = baseline_img(img)
    # rgbimg has to be after baseline_img()
    rgbimg = cv2.imread('tmp.png')
    line_with_most_white_pixels = find_whitest_line(img)  
    First_white_pixel,Last_white_pixel,Approx_centre = points(img[line_with_most_white_pixels])
    Approx_centre = int((First_white_pixel + Last_white_pixel)/2)
    print(f'{First_white_pixel= } {Last_white_pixel= } {Approx_centre= }')
    rgbimg[line_with_most_white_pixels] = [[255,0,0]] * len(rgbimg[0])
    for i in range(len(rgbimg)):
        rgbimg[i][Approx_centre] = [255,0,0]
    plt.imshow(rgbimg,cmap = 'gray')
    plt.show()
    os.remove('tmp.png')

if __name__ == "__main__":
    main()
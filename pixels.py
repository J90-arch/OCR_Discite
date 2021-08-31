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
def find_line_with_most_white_pixels(img):
    number_of_white_pixels = 0
    for i in range(len(img)//4,len(img)//4*3):
        num = np.sum(img[i] == 255)
        if num >= number_of_white_pixels:
            number_of_white_pixels = num
            line_with_most_white_pixels = i
    return line_with_most_white_pixels

# Find first and last pixel in that line within 1/20 of image length to other white pixels
def points(line_with_most_white_pixels):
    for i in range(len(line_with_most_white_pixels)//20 , len(line_with_most_white_pixels)):
        if line_with_most_white_pixels[i] == 255 and 255 in line_with_most_white_pixels[i:i+len(line_with_most_white_pixels)//20]:
            First_white_pixel = i
            break
    for i in range(First_white_pixel , len(line_with_most_white_pixels)):
        if 255 not in line_with_most_white_pixels[i:i+len(line_with_most_white_pixels)//20]:
            Last_white_pixel = i-1
            break
    return First_white_pixel,Last_white_pixel

def find_curve(img, First_white_pixel, Last_white_pixel, line_with_most_white_pixels):
    # y=-c((x+2)^2-a*log(x+2))+b
    counts = []
    count = 0
    A,B,C=0,0,0
    for x in range(First_white_pixel+1, Last_white_pixel):
        for y in range(line_with_most_white_pixels - len(img)//10, line_with_most_white_pixels):
            #print(x,y)
            
            a,b,c = calc_parabola_constants(First_white_pixel, line_with_most_white_pixels, x, y, Last_white_pixel, line_with_most_white_pixels)
            new_count = find_white_pixel_count_in_function(a,b,c,img)
            counts.append(count)
            if new_count > count:
                count = new_count
                A,B,C=a,b,c
            
            
            
    #print(counts)
    return A,B,C

def get_y_cord(a,b,c,x):
    #return int(c*((x+2)**2-a*np.log(x+2))+b)
    #print(f"x {x} y {a*(x**2) - b*x +c}")
    return int(a*(x**2) - b*x +c)

def find_white_pixel_count_in_function(a,b,c,img):
    X = len(img[0])
    Y = len(img)
    y_cords = [get_y_cord(a,b,c,i) for i in range(len(img))]
    x_cord = 0
    count = 0
    for item in y_cords:
        #print(f"y {item} | x {x_cord}")
        try:
            if img[item][x_cord] == 255:
                count+=1
        except IndexError:
            #print(f"dx{X-x_cord} | dy {Y-item}")
            pass
        x_cord+=1
    print(f'count {count}')
    return count

def calc_parabola_constants(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    print(f"DENOM {denom} x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}, x3 {x3}, y3 {y3}")
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    return A,B,C
        


def main():
    PATH = "test.jpg"
    img = cv2.imread(PATH, 0)
    img = baseline_img(img)
    # rgbimg has to be after baseline_img()
    rgbimg = cv2.imread('tmp.png')
    line_with_most_white_pixels = find_line_with_most_white_pixels(img)  
    First_white_pixel,Last_white_pixel = points(img[line_with_most_white_pixels])
    print(f'{First_white_pixel= } {Last_white_pixel= }')
    



    #for i in range(line_with_most_white_pixels-7 , line_with_most_white_pixels+8):
    #    rgbimg[i][First_white_pixel] = [0,255,0]
    #    rgbimg[i][Last_white_pixel] = [0,255,0]

    #rgbimg[line_with_most_white_pixels][First_white_pixel-7:First_white_pixel+8] = [[0,255,0]]*15
    #rgbimg[line_with_most_white_pixels][Last_white_pixel-7:Last_white_pixel+8] = [[0,255,0]]*15
    
    #rgbimg[line_with_most_white_pixels] = [[255,0,0]] * len(rgbimg[0])
    
    a,b,c = find_curve(img, First_white_pixel,Last_white_pixel,line_with_most_white_pixels)

    y_cords = [get_y_cord(a,b,c,i) for i in range(len(img[0]))]
    x_cords = [i for i in range(len(img[0]))]

    for x, y in zip(x_cords, y_cords):
        try:
            rgbimg[y][x] = [0,0,255]
        except:
            break
    plt.imshow(rgbimg,cmap = 'gray')
    plt.show()
    os.remove('tmp.png')
    
    print(find_white_pixel_count_in_function(a,b,c,img))

if __name__ == "__main__":
    main()
import os
import math
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
def points(line_with_most_white_pixels):#array of line of image
    for i in range(len(line_with_most_white_pixels)//20 , len(line_with_most_white_pixels)):
        if line_with_most_white_pixels[i] == 255 and 255 in line_with_most_white_pixels[i:i+len(line_with_most_white_pixels)//20]:
            First_white_pixel = i
            break
    for i in range(First_white_pixel , len(line_with_most_white_pixels)):
        if 255 not in line_with_most_white_pixels[i:i+len(line_with_most_white_pixels)//20]:
            Last_white_pixel = i-1
            break
    return First_white_pixel,Last_white_pixel


def P0_P2(line_with_most_white_pixels, img):
    X = len(img[0])
    Y = len(img)
    for i in range(line_with_most_white_pixels, line_with_most_white_pixels+Y//5):
        #if 255 in [a[X//15] for a in img[i-5: i+5]] and 255 not in [a[X//15] for a in img[i-len(line_with_most_white_pixels)//25:i]]:
        if 255 in img[i][(X//10)-10:(X//10)+10]:
            P0_y = i
            break
    for i in range(line_with_most_white_pixels, line_with_most_white_pixels+Y//5):
        #if 255 in [a[-X//15] for a in img[i-5: i+5]] and 255 not in [a[-X//15] for a in img[i-len(line_with_most_white_pixels)//25:i]]:
        if 255 in img[i][(5*X//8)-10:(5*X//8)+10] :
            P2_y = i
            break

    P0, P2 = np.array([[X//10, P0_y],[5*X//8, P2_y]])
    return P0, P2



def find_curve(img, First_white_pixel, Last_white_pixel, line_with_most_white_pixels, P0, P2):
    count = 0
    best_P1 = np.array([0, 0])
    for x in range(First_white_pixel+1, Last_white_pixel, 10):#delete 10 later
        for y in range(line_with_most_white_pixels - (len(img)//10), line_with_most_white_pixels, 10):
            print(f'x {x-First_white_pixel} out of {Last_white_pixel-First_white_pixel} y {y-line_with_most_white_pixels + (len(img)//10)} out of {(len(img)//10)}')
            P1 = np.array([x, y])
            new_count = find_white_pixel_count_in_function(img, P0, P1, P2)
            if new_count > count:
                count = new_count
                best_P1 = P1

    #print(counts)
    return best_P1



def find_white_pixel_count_in_function(img, P0, P1, P2):
    P = lambda t: (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2
    x_length = len(img[0])
    coords = np.array([P(t) for t in np.linspace(0, 1, x_length)])
    x_cords, y_cords = coords[:,0], coords[:,1]
    x_cords = [int(math.floor(a)) for a in x_cords]
    y_cords = [int(math.floor(a)) for a in y_cords]
    '''
    rgbimg = img
    for x, y in zip(x_cords, y_cords):
        try:
            rgbimg[y][x] = [0,255,255]
        except:
            break
    plt.imshow(rgbimg,cmap = 'gray')
    plt.show()
    '''
    tmp_list = []
    count = 0
    for x, y in zip(x_cords, y_cords):
        try:
            tmp_list.append(img[y][x])
            if img[y][x] == 255:
                count+=1
        except IndexError:
            pass
    #print(f'list {tmp_list}')
    #print(f'count {count}')
    return count

        


def main():
    # define bezier curve
    P = lambda t: (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2
    PATH = "test.jpg"
    og_img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    img = baseline_img(og_img)
    # rgbimg has to be after baseline_img()
    rgbimg = cv2.imread('tmp.png')
    line_with_most_white_pixels = find_line_with_most_white_pixels(img)  
    First_white_pixel, Last_white_pixel = points(img[line_with_most_white_pixels])
    
    #for i in range(line_with_most_white_pixels-7 , line_with_most_white_pixels+8):
    #    rgbimg[i][First_white_pixel] = [0,255,0]
    #    rgbimg[i][Last_white_pixel] = [0,255,0]
    #rgbimg[line_with_most_white_pixels][First_white_pixel-7:First_white_pixel+8] = [[0,255,0]]*15
    #rgbimg[line_with_most_white_pixels][Last_white_pixel-7:Last_white_pixel+8] = [[0,255,0]]*15
    #rgbimg[line_with_most_white_pixels] = [[255,0,0]] * len(rgbimg[0])
    P0, P2 = P0_P2(line_with_most_white_pixels, img)
    P1 = find_curve(img, First_white_pixel,Last_white_pixel,line_with_most_white_pixels, P0, P2)

    P = lambda t: (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2
    x_length = len(img[0])
    cords = np.array([P(t) for t in np.linspace(-0.5, 1.5, 2*x_length)])
    x_cords, y_cords = cords[:,0], cords[:,1]
    x_cords = [int(math.floor(a)) for a in x_cords]
    y_cords = [int(math.floor(a)) for a in y_cords]

    #for x, y in zip(x_cords, y_cords):
    #    try:
    #        rgbimg[y][x] = [255,255,255]
    #    except:
    #        break
    
    first_x, last_x = x_cords.index(0), x_cords.index(len(img[0]))

    x_cords = x_cords[first_x:last_x]
    y_cords = y_cords[first_x:last_x]


    

    show_img = np.array(cv2.resize(og_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    i = 0
    while(i < show_img.shape[1]-1):
        if x_cords[i+1] == x_cords[i]:
            del x_cords[i+1]
            del y_cords[i+1]
        else:
            i+=1
    if x_cords[-1] == x_cords[-2]:
        del x_cords[-1]
        del y_cords[-1]
    if x_cords[-1] == show_img.shape[1]:
        del x_cords[-1]
        del y_cords[-1]

    plt.imshow(show_img, cmap='gray')
    plt.plot(*P0, 'r.')
    plt.plot(*P1, 'r.')
    plt.plot(*P2, 'r.')
    plt.text(*P0, "P0",color="red")
    plt.text(*P1, "P1",color="red")
    plt.text(*P2, "P2",color="red")
    #plt.plot([0, 3000], [line_with_most_white_pixels, line_with_most_white_pixels])
    
    os.remove('tmp.png')
    plt.plot(x_cords, y_cords, 'r')
    plt.show()

    print(y_cords, show_img.shape)
    for a,b in zip(x_cords[1:], x_cords[:-1]):
        if a-b !=1:
            print(a-b)
    
    max_H = max(y_cords)
    print(f'max {max_H}')
    print(show_img)
    np.rot90(show_img)
    img_output = np.zeros([show_img.shape[0]+max_H,show_img.shape[1]], dtype=img.dtype)
    print(f'shape{img_output.shape} old {show_img.shape}')
    for y in range(show_img.shape[0]):
        for x in range(show_img.shape[1]):
            #print(f'h {y_cords[x]}')
            img_output[max_H + y - y_cords[x]][x] = show_img[y][x]
    #np.rot90[img_output, 3]
    plt.imshow(img_output)
    plt.show()
    #img_output = cv2.adaptiveThreshold(img_output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7)
    kernel = np.ones((1, 1), np.uint8)
    show_img = cv2.dilate(show_img, kernel, iterations=1)
    show_img = cv2.erode(show_img, kernel, iterations=1)
    show_img = cv2.GaussianBlur(show_img, (5, 5), 0)
    cv2.imwrite("img_output.jpg", img_output)



if __name__ == "__main__":
    main()
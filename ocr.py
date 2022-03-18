#!/usr/bin/env python3
import os
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import pytesseract
import re

###Code currently is held together by ducktape, so expect it to break
def baseline_img(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.threshold(img, 64, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("test-mono.jpg", img)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7) #imgf contains Binary image
    #img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7) 
    
    img = np.array(img, int)
    img = np.rot90(img)#rotate image 90 degrees so columns are one list each
    # we need to use negative j, because we want to evaluate pixels from bottom to the top
    l = img.shape[1]//15
    edges = np.zeros(img.shape, dtype = int)
    for i in range(len(img)):
        for j in range(l -1, len(img[0])):
            if img[i][-j] == 0 and 0 not in img[i][-j+1:-j+l]:
                edges[i][-j] = 255
                j+=l
        
    edges = np.rot90(edges, 3)
    #remove single white pixels
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1] -1):
            if edges[i][j] == 255:
                if edges[i][j-1] == 0 and edges[i][j+1] == 0 :
                    edges[i][j] = 0
    cv2.imwrite('tmp.png', edges)
    return edges

def F(x, a, m, n):
    return int(a*(x-m)**2 +n)
    
# Find a horizontal line with most pixels in middle half of the image
def find_line_with_most_white_pixels(img):
    number_of_white_pixels = 0
    for i in range(len(img)//4,len(img)//4*3):
    #for i in range(len(img)):
        num = np.sum(img[i] == 255)
        if num > number_of_white_pixels:
            #print(f'line_with_most_white_pixels {number_of_white_pixels} => {num} which is {i}')
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

def find_white_pixel_count_in_function(img, a, m, n):
    x_length = len(img[0])
    x_cords = [F(x,a,m,n) for x in range(0, len(img[0]))]
    y_cords = [i for i in range(0, len(img[0]))]
    count = 0
    for x, y in zip(x_cords, y_cords):
        try:
            if img[y-1][x] == 255:
                count+=1
            if img[y][x] == 255:
                count+=1
            if img[y+1][x] == 255:
                count+=1
        except IndexError:
            pass
    return count

def find_curve(img, line_with_most_white_pixels, First_white_pixel, Last_white_pixel):
    best_cnt =0
    best_a, best_n = 0,0
    m = (First_white_pixel+Last_white_pixel)/2
    a = 0
    n = 0
    P = lambda x: a*(x-m)**2 + n
    #max_a = min(((len(img)-1)-n)/(m**2), ((len(img)-1)-n)/((len(img[0])-1-m)**2))
    max_n = line_with_most_white_pixels
    n_list = np.arange(0.0, max_n, 0.01)
    cnt_since_last_change = 0
    for n in n_list[::-1]:
        cnt_since_last_change+=1
        new_a = (line_with_most_white_pixels - n)/((((Last_white_pixel-First_white_pixel)**2)/4))
        if F(0,new_a,m,n)>len(img) or F(len(img[0]), new_a,m,n)>len(img):
            pass
        else: 
            a = new_a
            cnt = find_white_pixel_count_in_function(img, a, m, n)
            if cnt > best_cnt and cnt_since_last_change<20:
                cnt_since_last_change = 0
                #print(f'best_count {cnt} => {best_cnt}, a {a}')
                best_n = n
                best_a = a
                best_cnt = cnt
    return best_a, m, best_n
    
def write_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def image_to_string(lan, img):
    #img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.threshold(img, 56, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("ocrimage.png", img)
    text = pytesseract.image_to_string(img, lang=lan)
    return text

def fix_text(text):
	text = re.sub("\n\s+", "\n", text)
	text = re.sub("-\n", "", text)
	text = re.sub("\|", "", text)
	return text

def main():
    ap = argparse.ArgumentParser(description="Program to preprocess images")
    ap.add_argument("-i", "--image", required=True, help="path to input image file")
    ap.add_argument("-f", "--file", required=False, help="file to write text to")
    ap.add_argument("-l", "--lang", required=False, help="language to read", default="eng")
    ap.add_argument("-d", "--debug", action="store_true", help="debug")
    args = vars(ap.parse_args())
    PATH = args["image"]
    DEBUG = bool(args["debug"])
    og_img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    img = baseline_img(og_img)
    line_with_most_white_pixels = find_line_with_most_white_pixels(img)  
    First_white_pixel, Last_white_pixel = points(img[line_with_most_white_pixels])
    a, m, n = find_curve(img, line_with_most_white_pixels, First_white_pixel, Last_white_pixel)
    show_img = np.array(cv2.resize(og_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))

    if DEBUG:
        plt.plot([i for i in range(0, len(img[0]))],[line_with_most_white_pixels for item in img[0]], 'r')
        plt.plot([First_white_pixel for item in img[0]],[i for i in range(0, len(img[0]))],'r')
        plt.plot([Last_white_pixel for item in img[0]],[i for i in range(0, len(img[0]))],'r')
        x_cords = [i for i in range(0, len(img[0]))]
        y_cords = [F(x,a,m,n) for x in range(0, len(img[0]))]
        plt.plot(x_cords,y_cords,'b')
        plt.imshow(show_img, cmap='gray')
        
        plt.show()
    ############# F(x, a, m, n)
    output_height_increase = max(F(len(show_img[0]),a,m,n), F(0,a,m,n))-F(m,a,m,n)
    output_height = int(len(show_img)) + output_height_increase
    output_length = int(len(show_img[0]))
    img_output = []
    #print(f'{len(img_output) = } {len(img_output[0]) = }')
    mono_img = cv2.imread("test-mono.jpg")
    if not DEBUG:
        os.remove('tmp.png')
        os.remove("test-mono.jpg")
    img_output = [[[0, 0, 0]]*output_length for i in range(output_height)]
    img_from = mono_img
    A = +F(n,a,m,n) + output_height_increase
    for i in range(int(len(img_from[0]))):
        d = int(A-F(i,a,m,n))
        for j in range(int(len(img_from))):
            try:
                img_output[j+d][i] = img_from[j][i]
            except IndexError:
                pass
    cv2.imwrite("img_output.jpg", np.array(img_output))
    ocr_img = cv2.imread("img_output.jpg")
    #os.remove("img_output.jpg")
    try:
        text = image_to_string(args["lang"], ocr_img)
        if args["file"]:
            write_file(fix_text(text), args["file"])
        else:
            print(text)
    except pytesseract.pytesseract.TesseractError:
        print(f'No such language installed "{args["lang"]}" or bad language format (should be three letters)')

if __name__ == "__main__":
    main()
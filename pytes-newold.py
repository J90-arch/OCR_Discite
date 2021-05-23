import sys
import re
import cv2
import numpy as np
try:
    import pytesseract
except ImportError:
    print('no pytesseract detected \n aborting...')
    quit()


usage ={
    '' : 'use -h for help',
    '-h' : 'usage : [mode] [language] [image] [file] \n use -h + [part] for more help \n use exit to terminate program \n example: -r eng test.png text.txt',
    'mode' : '-v visual mode \n-r writing mode',
    'language' : 'three letters or full name ex: eng',
    'image' : 'path or name of the image',
    'file' : 'file name or path to write text to'
}


def image_to_string(lan, image):
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.threshold(img, 128, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(img, lang=lan)
    text = re.sub("\n\s+", "\n", text)
    text = re.sub("-\n", "", text)
    text = re.sub("|", "", text)
    return text

def main():
    print('   ____    _____  _____\n  / __ \  / ____||  __ \ \n | |  | || |     | |__) | \n | |  | || |     |  _  / \n | |__| || |____ | | \ \ \n  \____/  \_____||_|  \_\ ')
    print('___  _ ____ ____ _ ___ ____\n|  \ | [__  |    |  |  |___\n|__/ | ___] |___ |  |  |___')
    print('\nDeveloped by Jokubas Virsilas \nversion 2.0')
    while 1:
        inp = input('>>> ')
        inp_list = inp.split()
        if inp == 'exit':
            sys.exit()
        try:
            i = 0
            if inp_list[0] == '-h':
                if len(inp_list) == 1:
                    print(usage[inp_list[0]])
                else:
                    print(usage[inp_list[1]])
            else:
                if inp_list[0] == '-v' or inp_list[0] == '-r':
                    string = image_to_string(inp_list[1], inp_list[2])
                    if inp_list[0] == '-v':
                        print(string)
                    if inp_list[0] == '-r':
                        write_file(string, inp_list[3])
                else:
                    quit()

        except:
            print('something went wrong... check -h for usage')



if __name__ == '__main__':
    main()
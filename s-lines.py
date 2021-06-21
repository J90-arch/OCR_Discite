import pypillowfight
import cv2

PATH = r"C:\Users\jokub\Desktop\Work\git_rep_ocr\extras\tests\IMG_0771.jpg"
img = cv2.imread(PATH)

img_sl = pypillowfight.unpaper(img)
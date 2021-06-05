import cv2
import re
img = cv2.imread("test.png")
with open("detect.txt", "r") as f:
	s = f.readlines()
cood = []
for item in s:
	cood.append(re.sub("\n", "", item))
cord = []
for item in cood:
	cord.append(item.split())
#for i in range(len(cord)):
#	a = cord[i]
#	cord[i] = [a[2], a[3], a[0], a[1]]
print(cord)

for i in range(len(cord)):
	crop_img = img[int(cord[i][2]):int(cord[i][3]), int(cord[i][0]):int(cord[i][1])]
	name = "C:/Users/jokub/Documents/random code/pytesseract_stuff/non-visual/crops/" + str(i)+".png"	
	cv2.imwrite(name, crop_img)
#	cv2.imshow("cropped", crop_img)
#	cv2.waitKey(0)
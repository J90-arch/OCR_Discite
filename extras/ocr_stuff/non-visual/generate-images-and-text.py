import easyocr
import cv2
language = 'lt'
img = "test.png"
text = []
reader = easyocr.Reader([language])
result = reader.detect(img)
for x in result[0]:
	str = reader.readtext(img[x[2]:x[3], x[0]:x[1]], detail = 0)
	text.append(str[0])
with open("text.txt", "w", encoding="utf-8") as f:
	f.write(" ".join(text))
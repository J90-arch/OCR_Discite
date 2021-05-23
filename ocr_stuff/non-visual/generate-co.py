import easyocr

language = 'lt'
img = 'test.png'

reader = easyocr.Reader([language])
result = reader.detect(img)
#print(result)
x = result[0]
for i in range(len(x)):
	x[i] = " ".join([str(z) for z in x[i]])+"\n"
	print(x[i])
print(x)
with open("detect.txt", "w", encoding="utf-8") as f:
	f.writelines(x)
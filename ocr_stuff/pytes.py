def main():
	try:
	    from PIL import Image
	except ImportError:
	    import Image
	import pytesseract
	import sys
	import re
	import cv2
	import numpy as np
	try:	
		if sys.argv[1] == '-h':
			print('usage:\npython pytes.py [language] [Image file] [Output file]\nexample:\npython pytes.py lit test.png text.txt\nAvaliable languages:')
			print(pytesseract.get_languages(config=''))
		else:
			lan = sys.argv[1]
			im = sys.argv[2]
			fil = sys.argv[3]
			img = cv2.imread(im)
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
			with open(fil, "w", encoding="utf-8") as f:
				f.write(text)
	except LookupError:
		print('LookupError')
	except TypeError:
		print('TypeError')
	except cv2.error:
		print('cv2.error')
if __name__ == '__main__':
	main()
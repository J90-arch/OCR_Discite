Uses:
numpy
matplotlib
cv2
PIL
easyocr

Run in order:
gray.py		(optional)
	creates black and white image for fourier to work with, but fourier can work with original images too.
	It should try to help with background noise. takes in test.png
fourier.py or fourier-visual.py(for troubleshooting)	(optional)
	rotates test.png and saves as rgray.png by using fourier magic.
generate-text.py	(required)
	takes in test.png and creates text.txt of output text.





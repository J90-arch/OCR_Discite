def get_string(img_path):
	# Read image using opencv
	img = cv2.imread(img_path)

	# Extract the file name without the file extension
	file_name = os.path.basename(img_path).split('.')[0]
	file_name = file_name.split()[0]

	# Create a directory for outputs
	output_path = os.path.join(output_dir, file_name)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	# Rescale the image, if needed.
	img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

	# Convert to gray
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply dilation and erosion to remove some noise
	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)
	# Apply blur to smooth out the edges
	img = cv2.GaussianBlur(img, (5, 5), 0)

	# Apply threshold to get image with only b&w (binarization)
	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# Save the filtered image in the output directory
	save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
	cv2.imwrite(save_path, img)

	# Recognize text with tesseract for python
	result = pytesseract.image_to_string(img, lang="lit")
	return result
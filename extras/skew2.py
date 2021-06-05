import cv2
image = cv2.imread(r"rotated.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
otsu_thresh, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel_size = 4
ksize=(kernel_size, kernel_size)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
thresh_filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
nonZeroCoordinates = cv2.findNonZero(thresh_filtered)
imageCopy = image.copy()
for pt in nonZeroCoordinates:
    imageCopy = cv2.circle(imageCopy, (pt[0][0], pt[0][1]), 1, (255, 0, 0))
box = cv2.minAreaRect(nonZeroCoordinates)
boxPts = cv2.boxPoints(box)
for i in range(4):
    pt1 = (boxPts[i][0], boxPts[i][1])
    pt2 = (boxPts[(i+1)%4][0], boxPts[(i+1)%4][1])
    cv2.line(imageCopy, pt1, pt2, (0,255,0), 2, cv2.LINE_AA);

angle = box[2]
print(angle)
if(angle < -45):
    angle = angle
else:
    angle = angle -90


h, w, c = image.shape
scale = 1.
center = (w/2., h/2.)
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = image.copy()
rotated = cv2.warpAffine(image, M, (w, h), rotated, cv2.INTER_CUBIC, cv2.BORDER_REPLICATE )
cv2.imshow('rot', rotated)
cv2.waitKey()
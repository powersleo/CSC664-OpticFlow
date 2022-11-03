import numpy as np
import cv2
import time

video = cv2.VideoCapture('zebraFish\X265-Crf15-1.mp4')
ret, frame = video.read()
data = np.asarray(frame, dtype="int32")

gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
#need threshold value?
edges = cv2.Canny(frame,200,300 )
imagex = data.shape[0]
imagey = data.shape[1]
print(imagex, imagey)
frameWidthx = (int)(imagex/8)
frameWidthy = (int)(imagey/12)
borderOffset = 10
# cropped = gray_image[borderOffset:frameWidthx ,borderOffset:frameWidthx ]

for x in range(1,8):
    for y in range(1,12):
        frameNumberX = x
        frameNumberY = y
        imageName = "cell " + (str)(x) + ", " + (str)(y)
        cv2.imshow(imageName, gray_image[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
frameNumberX = 1
frameNumberY = 5
croppedCell = gray_image[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]

# print(edges)
#print array as int
# for x in np.nditer(data):
#     print(x, end=" ")

# print(data[0,0,0], data[0,0,1], data[0,0,2])
# cv2.imshow("test", cropped)
# cv2.imshow("test2",cropped2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("edges.jpeg", edges)
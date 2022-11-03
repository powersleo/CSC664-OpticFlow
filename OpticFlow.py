import numpy as np
import cv2
import time

video = cv2.VideoCapture('zebraFish\X265-Crf15-1.mp4')
ret, frame = video.read()
data = np.asarray(frame, dtype="int32")

#need threshold value?
#edges = cv2.Canny(frame, )

#print array as int
for x in np.nditer(data):
    print(x, end=" ")
print(data.shape)
cv2.imwrite("test.jpeg", data )
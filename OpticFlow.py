import numpy as np
import cv2
import time
# from scipy import signal1 
# video = cv2.VideoCapture('zebraFish\X265-Crf15-1.mp4')
# ret, frame = video.read()
# data = np.asarray(frame, dtype="int32")


# #set image to grey Scale
# gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #need threshold value?
# edges = cv2.Canny(frame,200,300 )

def OpticFlow(frame1, frame2):
    #convert frames to optic flow
    frame1Array = np.array(frame1)
    frame2Array = np.array(frame2)
    # set image shape
    imageResolution = np.shape(frame1Array)
    w = int(2/2)
    cornerList = cv2.goodFeaturesToTrack(frame1Array, 10000, 0.1, 0.1)

    #apply gaussianBlur
    frame1_blur = cv2.GaussianBlur(frame1Array,(3,3), 0)
    frame2_blur = cv2.GaussianBlur(frame2Array,(3,3), 0)

    convolveFilterX = np.array([[-1, 1], [-1, 1]])
    # [-1,1]
    # [-1,1]

    convolveFilterY = np.array([[-1, -1], [1, 1]])
    # [-1,-1]
    # [1 , 1]

    convolveFilterT = np.array([[1, 1], [1, 1]])
    # [1,1]
    # [1,1]
    

    #calculate the partial derivative of the images
    frameX = cv2.filter2D(frame1_blur, -1, convolveFilterX)
    frameY = cv2.filter2D(frame1_blur, -1, convolveFilterY)
    frameT = cv2.filter2D(frame2_blur, -1, convolveFilterT) - cv2.filter2D(frame1_blur, -1, convolveFilterT)
    cv2.imshow("original", frame1)
    cv2.imshow("framex", frameX)
    cv2.imshow("framey", frameY)
    cv2.imshow("framet", frameT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    u = np.zeros(frame1_blur.shape)
    v = np.zeros(frame1_blur.shape)

    for corners in cornerList: 
        i, j = corners.ravel()
        i, j = int(i),int(j)
        # print(i,j)
        I_x = frameX[i-w:i+w+1, j-w:j+w+1].flatten()
        I_y = frameY[i-w:i+w+1, j-w:j+w+1].flatten()
        I_t = frameT[i-w:i+w+1, j-w:j+w+1].flatten()
        print(i,j,"\n",I_x,"\n",I_y, "\n",I_t,"\n")
        b = np.reshape(I_t, (I_t.shape[0],1))
        A = np.vstack((I_x, I_y)).T

        U = np.matmul(np.linalg.pinv(A), b) 

        u[i,j] = U[0][0]
        v[i,j] = U[1][0]
    return (u,v)
       


# imagex = data.shape[0]
# imagey = data.shape[1]
# print(imagex, imagey)
# frameWidthx = (int)(imagex/8)
# frameWidthy = (int)(imagey/12)
# borderOffset = 10

image1 = cv2.imread("zebraFish/frames_grey/0001.png",1)
image2 = cv2.imread("zebraFish/frames_grey/0003.png",1)

gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# print(OpticFlow(gray_image, gray_image2))
# cropped = gray_image[borderOffset:frameWidthx ,borderOffset:frameWidthx]

for x in np.nditer(OpticFlow(gray_image, gray_image2)):
    if(x[0] or x[1]):
        print(x[0], x[1], end=" \n")

#iterate through cells
#press any key to go to next cell

# for x in range(1,9):
#     for y in range(1,13):
#         frameNumberX = x
#         frameNumberY = y
#         imageName = "cell " + (str)(x) + ", " + (str)(y)
#         cv2.imshow(imageName, gray_image[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# frameNumberX = 1
# frameNumberY = 5
# croppedCell = gray_image[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]

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
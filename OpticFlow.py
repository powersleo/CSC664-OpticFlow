import numpy as np
import cv2
import time
import math
# from scipy import signal1 
# video = cv2.VideoCapture('zebraFish\X265-Crf15-1.mp4')
# ret, frame = video.read()
# data = np.asarray(frame, dtype="int32")


# #set image to grey Scale
# gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #need threshold value?
# edges = cv2.Canny(frame,200,300 )

def unitVector(vector1, vector2):
    x = vector2[0] - vector1[0]
    y = vector2[1] - vector1[1]
    magnitude = math.sqrt(x * x + y * y)
    if(magnitude != 0):
        return(x/magnitude, y/magnitude)
    else:
        return (0,0)
    
def OpticFlow(frame1, frame2, cornerList):
    

    # # roberts derivative filter
    # convolveFilterX = np.array([[-1, 1], [-1, 1]])
    # # [-1,1]
    # # [-1,1]

    # convolveFilterY = np.array([[1, 1], [-1, -1]]) 
    # # [-1,-1]
    # # [1 , 1]

    # convolveFilterT = np.array([[1, 1], [1, 1]])
    # # [1,1]
    # # [1,1]
    # convolveFilterTi = np.array([[-1, -1], [-1, -1]])
    # # [1,1]
    # # [1,1]
    
    # roberts derivative filter
    convolveFilterX = np.array([[0, 1], [-1, 0]])
    # [-1,1]
    # [-1,1]

    convolveFilterY = np.array([[1, 0], [0, -1]]) 
    # [-1,-1]
    # [1 , 1]

    convolveFilterT = np.array([[1, 1], [1, 1]])
    # [1,1]
    # [1,1]
    convolveFilterTi = np.array([[-1, -1], [-1, -1]])
    # [1,1]
    # [1,1]
    
    # using 3x3 kernel
    # prewitt
    # convolveFilterX = np.array([[-1,0 ,1], [-1,0,1], [-1,0,1]])
    # # [-3,0 ,3]
    # # [-10,0,10]
    # # [-3,0,3]
    # convolveFilterY = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])
    # # [-3,10,3]    # # [0,0,0]
    # #[3,10,3]

    # convolveFilterT = np.array([[1, 1,1], [1, 1,1],[1,1,1]])
    # # [1,1,1]
    # # [1,1,1]
    # # [1,1,1]
    # convolveFilterTi = np.array([[-1, -1,-1], [-1,- 1,-1],[-1,-1,-1]])
    # # [1,1,1]
    # # [1,1,1]
    # # [1,1,1]

    # # using 3x3 kernel
    # #sobel
    # convolveFilterX = np.array([[-1,0 ,1], [-2,0,2], [-1,0,1]])
    # # [-3,0 ,3]
    # # [-10,0,10]
    # # [-3,0,3]
    # convolveFilterY = np.array([[1,2,1], [0,0,0],[-1,-2,-1]])
    # # [-3,10,3]
    # # [0,0,0]
    # #[3,10,3]

    # convolveFilterT = np.array([[1, 1,1], [1, 1,1],[1,1,1]])
    # # [1,1,1]
    # # [1,1,1]
    # # [1,1,1]
    # convolveFilterTi = np.array([[-1, -1,-1], [-1, -1,-1],[-1,-1,-1]])
    # # [1,1,1]
    # # [1,1,1]
    # # [1,1,1]


    convolveFilterX = convolveFilterX 
    convolveFilterY = convolveFilterY 
    convolveFilterT = convolveFilterT 
    
    # #image convolve for gradient 
    frameX = cv2.filter2D(frame1, -1, convolveFilterX) + cv2.filter2D(frame2, -1, convolveFilterX) 
    frameY = cv2.filter2D(frame1, -1, convolveFilterY) + cv2.filter2D(frame2, -1, convolveFilterY) 
    frameT = cv2.filter2D(frame2, -1, convolveFilterT) - cv2.filter2D(frame1, -1, convolveFilterT)
    # frameT = frame1 - frame2


    # cv2.imshow("original", frame1)
    cv2.imshow("framex", frameX)
    cv2.imshow("framey", frameY)
    cv2.imshow("framet", frameT)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)

    #size of frame to search
    w = int(3/2)

    for corners in cornerList: 
        i, j = corners.ravel()
        i, j = int(i),int(j)
        # print("this is i, j")
        # print(i,j)
        # print("\n")
         #determine size of search
        I_x = frameX[i-w:i+w + 1, j-w:j+w + 1].flatten()
        I_y = frameY[i-w:i+w + 1, j-w:j+w+1].flatten()
        I_t = frameT[i-w:i+w + 1, j-w:j+w+ 1].flatten()
        # print(i,j,"\n",I_x,"\n",I_y, "\n",I_t,"\n")
        b = np.reshape(I_t, (I_t.shape[0],1))
        
        
        
        A = np.vstack((I_x, I_y)).T
        
        AAT =np.dot(A, A.T)
        AATinv = np.linalg.pinv(AAT)
        ATB = np.dot(A.T, b)
        AATinvAt = np.dot(AATinv, A)
        U = np.dot(AATinvAt.T, b)
        
        # print(b,"\n", A,"\n")
        # print("this is U",U)
        # print("end of U")
        print(i,j)
        u[i-1,j-1] = U[0][0]
        v[i-1,j-1] = U[1][0]
    return (u,v)
       


# chrimagex = data.shape[0]
# imagey = data.shape[1]
# print(imagex, imagey)
# frameWidthx = (int)(imagex/8)
# frameWidthy = (int)(imagey/12)
# borderOffset = 10

# image1 = cv2.imread("zebraFish/frames_grey/0001.png",1)
# image2 = cv2.imread("zebraFish/frames_grey/0003.png",1)

#conver Images to greyscale
# gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# print(OpticFlow(gray_image, gray_image2))
# cropped = gray_image[borderOffset:frameWidthx ,borderOffset:frameWidthx]

#convert frames to np arrays
# frame1Array = np.array(gray_image)
# frame2Array = np.array(gray_image2)
# set image shape
# imageResolution = np.shape(frame1Array)


    
#apply gaussianBlur
# frame1_blur = cv2.GaussianBlur(frame1Array,(3,3), 0)
# frame2_blur = cv2.GaussianBlur(frame2Array,(3,3), 0)

#edge detection
#   Good point to iterate on
# cornerList = cv2.goodFeaturesToTrack(frame1_blur, 10000, 0.1, 0.1)
# cornerList2 = cv2.goodFeaturesToTrack(frame2_blur, 10000, 0.1, 0.1)

#return vectors of motion
# opticFlow = OpticFlow(gray_image, gray_image2,cornerList)
# mask = np.zeros_like(image1)
# print(cornerList)
# for corners in cornerList:
#     x = int(corners[0][0])
#     y = int(corners[0][1])
#     print(opticFlow[0][x,y])
#     print(opticFlow[1][x,y])
#     mask = cv2.line(mask, (x,y),(x + int(opticFlow[0][x,y] *10 ),y + int(opticFlow[1][x,y]*10 )),(255,0,0),1)
#     img = cv2.add(image1, mask)

# cv2.imshow("image 1",img)
# cv2.imwrite("image1.jpeg",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# lk_params = dict( winSize = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                               10, 0.03))
# p0 = cv2.goodFeaturesToTrack(gray_image, mask = None,
#                              **feature_params)
  

# p1, st, err = cv2.calcOpticalFlowPyrLK(gray_image,
#                                            gray_image2,
#                                            p0, None,
#                                            **lk_params)
# print("\n",p1,"\n", st,"\n")
# # for corners in 


# for x in OpticFlow(gray_image, gray_image2,cornerList):
#     # if(x[0] or x[1]):
#     print(type(x), end=" \n")

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

video = cv2.VideoCapture('zebraFish/X265-Crf15-1.mp4')
ret, frameInitial = video.read()
borderOffset = 10
imagex = frameInitial.shape[0]
imagey = frameInitial.shape[1]
frameWidthx = (int)(imagex/8)
frameWidthy = (int)(imagey/12)
frameNumberX = 1
frameNumberY = 1
frameInitial = frameInitial[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]
frameInitialResize = cv2.resize(frameInitial,(frameInitial.shape[0] *3, frameInitial.shape[1]* 3))

frameInitialGrey = cv2.cvtColor(frameInitialResize, cv2.COLOR_BGR2GRAY)
frameInitialArray = np.array(frameInitialGrey)
frameInitial_blur = cv2.GaussianBlur(frameInitialArray,(3,3), 0)
cornerList = cv2.goodFeaturesToTrack(frameInitial_blur, 10, 0.01, 10)

mask = np.zeros_like(frameInitialResize)
writeFrame = frameInitialResize
i = 0.0
while(ret):
    color = np.random.randint(0, 255, [100, 3])
    ret, frameDelta = video.read()
    frameDelta = frameDelta[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]
    frameDeltaResize = cv2.resize(frameDelta,(frameDelta.shape[0] *3, frameDelta.shape[1]* 3))
    frameDeltaGrey = cv2.cvtColor(frameDeltaResize, cv2.COLOR_BGR2GRAY)
    print("image size",frameDelta.shape, "image size:\n")
    frameDeltaArray = np.array(frameDeltaGrey)
    frameDelta_blur = cv2.GaussianBlur(frameDeltaArray,(3,3), 0)
    opticFlow = OpticFlow(frameInitial_blur, frameDelta_blur,cornerList)
    for corners in cornerList:
        x = int(corners[0][0])
        y = int(corners[0][1])
        # print(opticFlow[0][x,y])
        # print(opticFlow[1][x,y])
        # print((x,y))
        # print(opticFlow[0][x,y] ),y + int(opticFlow[1][x,y])
        print(unitVector((x-1,y-1), (x-1 + (opticFlow[0][x-1,y-1]  ),y-1 + (opticFlow[1][x-1,y-1]))))
        normalizedVector = unitVector((x-1,y-1), (x-1 + int(opticFlow[0][x-1,y-1]  ),y-1 + int(opticFlow[1][x-1,y-1])))
        
        
        # print(type(normalizedVector))
        if((x-1,y-1) != (x-1 + normalizedVector[0],y-1 + normalizedVector[1])):
            print((x-1,y-1),(int(x-1 + (normalizedVector[0]*3)),int(y-1 + (normalizedVector[1]*3))))
            mask = cv2.arrowedLine(mask,(int(x-1 + (normalizedVector[0]*7)),int(y-1 + (normalizedVector[1]*7))), (x-1,y-1),(255,0,0),1, tipLength=.25)
        if(i % 10 == 0 ):
            mask = np.zeros_like(frameInitialResize)
        print(i % 100)

        img = cv2.add(frameDeltaResize, mask)    
        img = cv2.resize(img, (1000, 1000))
    cv2.imshow('frame', img)
    k = cv2.waitKey(333)
    if k == 27:
        break
    i += 1
    frameInitial_blur = frameDelta_blur
    cornerList = cv2.goodFeaturesToTrack(frameDeltaGrey, 10, 0.01, 10)
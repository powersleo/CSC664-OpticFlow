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

def gridView(image1, searchSize):
    gridArray = []
    imageShape = image1.shape
    for i in range(0,imageShape[0],searchSize):
        for j in range(0,imageShape[1],searchSize):
            # grid = np.array([(i,j)], axis=2)
            gridArray += [(i,j)]
    return gridArray

def OpticFlow(frame1, frame2, cornerList):
    

    # # roberts derivative filter
    convolveFilterX = np.array([[-1, 1], [-1, 1]])
    # [-1,1]
    # [-1,1]

    convolveFilterY = np.array([[-1, -1], [1, 1]]) 
    # [-1,-1]
    # [1 , 1]

    convolveFilterT = np.array([[1, 1], [1, 1]])
    # [1,1]
    # [1,1]
    convolveFilterTi = np.array([[-1, -1], [-1, -1]])
    # [1,1]
    # [1,1]
    
    # roberts derivative filter
    # convolveFilterX = np.array([[0, 1], [-1, 0]])
    # # [-1,1]
    # # [-1,1]

    # convolveFilterY = np.array([[1, 0], [0, -1]]) 
    # # [-1,-1]
    # # [1 , 1]

    # convolveFilterT = np.array([[1, 1], [1, 1]])
    # # [1,1]
    # # [1,1]
    # convolveFilterTi = np.array([[-1, -1], [-1, -1]])
    # # [1,1]
    # # [1,1]
    
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


    convolveFilterX = convolveFilterX * 2 
    convolveFilterY = convolveFilterY * 2
    convolveFilterT = convolveFilterT 
    
    # #image convolve for gradient 
    frameX = cv2.filter2D(frame1, -1, convolveFilterX)
    frameY = cv2.filter2D(frame1, -1, convolveFilterY)
    frameT =  cv2.filter2D(frame1, -1, convolveFilterT) -  cv2.filter2D(frame2, -1, convolveFilterT)
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
    #because we search in 3X3 frame we want half of frame, change 3 value to change search parameter
    w = int(3/2)
    

    for corners in cornerList: 
        # print(corners)
        i, j = corners
        i, j = int(i),int(j)
        
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
        # print(i,j)
        if(i < frame1.shape[0] and j < frame1.shape[1] and i != 0 and j != 0):
            u[i,j] = U[1][0]
            v[i,j] = U[0][0]
            if(U[0][0] != 0 and U[1][0] != 0):
                print("Calculating at position: ")
                print(i,j)
                print("resultant vector = ",U[0][0],U[1][0] )
            
    return (u,v)
       



video = cv2.VideoCapture('zebraFish/X265-Crf15-1.mp4')
ret, frameInitial = video.read()
borderOffset = 10
imagex = frameInitial.shape[0]
imagey = frameInitial.shape[1]
frameWidthx = (int)(imagex/8)
frameWidthy = (int)(imagey/12)
frameNumberX = 1
frameNumberY = 1
gridOffset = 10
frameInitial = frameInitial[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]
frameInitialResize = cv2.resize(frameInitial,(frameInitial.shape[0] *3, frameInitial.shape[1]* 3))

frameInitialGrey = cv2.cvtColor(frameInitialResize, cv2.COLOR_BGR2GRAY)
frameInitialArray = np.array(frameInitialGrey)
frameInitial_blur = cv2.GaussianBlur(frameInitialArray,(3,3), 0)
# cornerList = cv2.goodFeaturesToTrack(frameInitial_blur, 100, 0.01, 5)
cornerList = gridView(frameInitial_blur, gridOffset)
# print("corner list", cornerList)

mask = np.zeros_like(frameInitialResize)
writeFrame = frameInitialResize
i = 0

color = np.random.randint(0, 255, (100, 3))

while(ret):
    ret, frameDelta = video.read()
    frameDelta = frameDelta[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx*frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset : frameWidthy*frameNumberY]
    
    #use to tweak frame scale factor
    frameScaleFactor = 3
    frameDeltaResize = cv2.resize(frameDelta,(frameDelta.shape[0] *frameScaleFactor, frameDelta.shape[1]* frameScaleFactor))
    frameDeltaGrey = cv2.cvtColor(frameDeltaResize, cv2.COLOR_BGR2GRAY)
    
    print("image size",frameDelta.shape, "image size:\n")
    frameDeltaArray = np.array(frameDeltaGrey)
    frameDelta_blur = cv2.GaussianBlur(frameDeltaArray,(3,3), 0)
    
    opticFlow = OpticFlow(frameInitial_blur, frameDelta_blur,cornerList)
    
    #factor to scale vector by
    vectorScaleFactor = 7
    colorIterator = 0
    # print("this is corner ", cornerList)
    for corners in cornerList:
        # x = int(corners[0][0])
        # y = int(corners[0][1])
        x,y = corners
        x,y = int(x),int(y)
        # print(opticFlow[0][x,y])
        # print(opticFlow[1][x,y])
        # print((x,y))
        # print(opticFlow[0][x,y] ),y + int(opticFlow[1][x,y])
        
        if(x < frameDeltaResize.shape[0] and y < frameDeltaResize.shape[1]):
            # print("writing arrow")
            # print(unitVector((x-1,y-1), (x-1 + (opticFlow[0][x-1,y-1]  ),y-1 + (opticFlow[1][x-1,y-1]))))
            normalizedVector = unitVector((x,y), (x + int(opticFlow[0][x,y]  ),y + int(opticFlow[1][x,y])))
            # print(type(normalizedVector))
            if((x,y) != (x - normalizedVector[0],y + normalizedVector[1])):
                # print((x,y),(int(x + (normalizedVector[0]*vectorScaleFactor)),int(y + (normalizedVector[1]*vectorScaleFactor))))
                # print()
                # print("writing arrow")
                mask = cv2.arrowedLine(mask,(y,x), (int(y + (normalizedVector[1]*vectorScaleFactor)),int(x - (normalizedVector[0]*vectorScaleFactor))),color[colorIterator].tolist(),1, tipLength=.25)
            # if(i % 2 == 0 ):
                
            # print(i % 100)
        
        img = cv2.add(frameDeltaResize, mask)
        img = cv2.resize(img, (1000, 1000))
    cv2.imshow('frame', img)
    fileName = "frames/frame" + str(i) + ".jpeg"
    cv2.imwrite(fileName, img)
    k = cv2.waitKey(1000)
    mask = np.zeros_like(frameInitialResize)    
    # cv2.destroyAllWindows()
    if k == 27:
        break
    i += 1
    colorIterator += 1
    frameInitial_blur = frameDelta_blur
    cornerList = gridView(frameDelta_blur, gridOffset)

import numpy as np
import cv2
import time
import math
import os
# todo
# use openCV to calculate the error estimation in our calculation


def calculateMagnitude(vector1, vector2):
    x = vector2[0] - vector1[0]
    y = vector2[1] - vector1[1]
    magnitude = math.sqrt(x * x + y * y)
    return magnitude


def unitVector(vector1, vector2):
    x = vector2[0] - vector1[0]
    y = vector2[1] - vector1[1]
    magnitude = calculateMagnitude(vector1, vector2)
    if (magnitude != 0):
        return (x/magnitude, y/magnitude)
    else:
        return (0, 0)


def gridView(image1, searchSize):
    gridArray = []
    imageShape = image1.shape
    for i in range(0, imageShape[0], searchSize):
        for j in range(0, imageShape[1], searchSize):
            gridArray += [(i, j)]
    return gridArray


def OpticFlow(frame1, frame2, cornerList, filterType=0, filterScaleX=1, filterScaleY=1, FilterScaleT=1):

    match filterType:
        case 0:
            # # roberts derivative filter
            convolveFilterX = np.array([[-1, 1], [-1, 1]])
            # [-1,1]
            # [-1,1]

            convolveFilterY = np.array([[1, 1], [-1, -1]])
            # [-1,-1]
            # [1 , 1]

            convolveFilterT = np.array([[1, 1], [1, 1]])
            # [1,1]
            # [1,1]
            convolveFilterTi = np.array([[-1, -1], [-1, -1]])
            # [1,1]
            # [1,1]
        case 1:
            #  derivative filter
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
        case 2:
            # using 3x3 kernel
            # prewitt
            convolveFilterX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            # [-3,0 ,3]
            # [-10,0,10]
            # [-3,0,3]
            convolveFilterY = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            # [-3,10,3]    # # [0,0,0]
            # [3,10,3]

            convolveFilterT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            # [1,1,1]
            # [1,1,1]
            # [1,1,1]
            convolveFilterTi = np.array(
                [[-1, -1, -1], [-1, - 1, -1], [-1, -1, -1]])
            # [1,1,1]
            # [1,1,1]
            # [1,1,1]
        case 3:
            # # using 3x3 kernel
            # #sobel
            convolveFilterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            # [-3,0 ,3]
            # [-10,0,10]
            # [-3,0,3]
            convolveFilterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            # [-3,10,3]
            # [0,0,0]
            # [3,10,3]
            convolveFilterT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            # [1,1,1]
            # [1,1,1]
            # [1,1,1]
            convolveFilterTi = np.array(
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
            # [1,1,1]
            # [1,1,1]
            # [1,1,1]
        case _:
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

    convolveFilterX = convolveFilterX * filterScaleX
    convolveFilterY = convolveFilterY * filterScaleY
    convolveFilterT = convolveFilterT * filterScaleT

    # #image convolutions
    frameX = cv2.filter2D(frame1, -1, convolveFilterX)
    frameY = cv2.filter2D(frame1, -1, convolveFilterY)
    frameT = cv2.filter2D(frame2, -1, convolveFilterT) - \
        cv2.filter2D(frame1, -1, convolveFilterT)
    # frameT = frame2 - frame1

    # cv2.imshow("original", frame1)

    # cv2.imshow("framex", frameX)
    # cv2.imshow("framey", frameY)
    # cv2.imshow("framet", frameT)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)

    # size of frame to search
    # because we search in 3X3 frame we want half of frame, change 3 value to change search parameter
    w = int(9/2)

    for corners in cornerList:
        # print(corners)
        i, j = corners
        i, j = int(i), int(j)

        # print("\n")
        # determine size of search
        I_x = frameX[i-w:i+w + 1, j-w:j+w + 1].flatten()
        I_y = frameY[i-w:i+w + 1, j-w:j+w+1].flatten()
        I_t = frameT[i-w:i+w + 1, j-w:j+w + 1].flatten()
        # print(i,j,"\n",I_x,"\n",I_y, "\n",I_t,"\n")
        b = np.reshape(I_t, (I_t.shape[0], 1))

        A = np.vstack((I_x, I_y)).T
        AAT = np.dot(A, A.T)
        AATinv = np.linalg.pinv(AAT)
        ATB = np.dot(A.T, b)
        AATinvAt = np.dot(AATinv, A)
        U = np.dot(AATinvAt.T, b)
        # print(b,"\n", A,"\n")
        # print("this is U",U)
        # print("end of U")
        # print(i,j)
        if (i < frame1.shape[1] and j < frame1.shape[0] and i != 0 and j != 0):
            u[j, i] = U[0][0]
            v[j, i] = U[1][0]
            # if(U[0][0] != 0 and U[1][0] != 0):
            # print("Calculating at position: ")
            # print(i,j)
            # print("resultant vector = ",U[0][0],U[1][0] )

    return (u, v)


# frame position of assay
frameNumberX = 1
frameNumberY = 1

# 12 X 8 grid

# size of grid for optic flow calculation
gridOffset = 5

# size to scale frame by
frameScaleFactor = 3

# write frames to a folder
writeFrames = True

# vector variables
isNormalizedVector = True
vectorScaleFactor = 7
showZeroVectors = False

# type of filter to convolve
filterType = 0

# used in optic flow calculation to scale image
filterScaleX = 1
filterScaleY = 1
filterScaleT = 1

# compare results to optic flow from openCV
compareResults = False


# discard vectors with magnitude smaller than this number
magnitudeConstraint = 20

# amount of time frame remains on screen
frameTime = 1


def calculateOpticFlowData(frameNumberX, frameNumberY, frameTime, frameToSample=0, gridOffset=5, frameScaleFactor=3, writeFrames=True, isNormalizedVector=True, vectorScaleFactor=7, showZeroVectors=False, filterType=0, filterScaleX=1, filterScaleY=1, filterScaleT=1, compareResults=False, magnitudeConstraint=20, view=True):
    video = cv2.VideoCapture('zebraFish/X265-Crf15-1.mp4')
    ret, frameInitial = video.read()
    borderOffset = 10
    imagex = frameInitial.shape[0]
    imagey = frameInitial.shape[1]
    frameWidthx = (int)(imagex/8)
    frameWidthy = (int)(imagey/12)
    frameInitial = frameInitial[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx *
                                frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset: frameWidthy*frameNumberY]
    frameInitialResize = cv2.resize(
        frameInitial, (frameInitial.shape[0] * 3, frameInitial.shape[1] * 3))
    frameInitialGrey = cv2.cvtColor(frameInitialResize, cv2.COLOR_BGR2GRAY)
    frameInitialArray = np.array(frameInitialGrey)
    frameInitial_blur = cv2.GaussianBlur(frameInitialArray, (3, 3), 0)

    # create positions to calculate optic flow for
    cornerList = gridView(frameInitial_blur, gridOffset)

    # mask used to draw arrows
    mask = np.zeros_like(frameInitialResize)

    # used to track frame number
    frameNumber = 0

    color = np.random.randint(0, 255, (100, 3))

    totalMagnitude = 0
    while (ret):
        ret, frameDelta = video.read()
        frameDelta = frameDelta[(frameWidthx) * (frameNumberX - 1) + borderOffset:frameWidthx *
                                frameNumberX, (frameWidthy) * (frameNumberY - 1) + borderOffset: frameWidthy*frameNumberY]

        # resize frame
        frameDeltaResize = cv2.resize(
            frameDelta, (frameDelta.shape[0] * frameScaleFactor, frameDelta.shape[1] * frameScaleFactor))
        frameDeltaGrey = cv2.cvtColor(frameDeltaResize, cv2.COLOR_BGR2GRAY)

        # print("image size",frameDelta.shape, "image size:\n")
        frameDeltaArray = np.array(frameDeltaGrey)
        frameDelta_blur = cv2.GaussianBlur(frameDeltaArray, (3, 3), 0)

        colorIterator = 0
        opticFlow = OpticFlow(frameInitial_blur, frameDelta_blur, cornerList,
                              filterType, filterScaleX, filterScaleY, filterScaleT)

        if (compareResults):
            p0 = cv2.goodFeaturesToTrack(frameInitial_blur, 100, mask=None,)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                frameInitial_blur, frameDelta_blur, p0, None)
            mask2 = np.zeros_like(frameInitialResize)
            print(p1)

        for corners in cornerList:
            x, y = corners
            x, y = int(x), int(y)
            if (x < frameDeltaResize.shape[0] and y < frameDeltaResize.shape[1]):
                if (isNormalizedVector):
                    normalizedVector = unitVector(
                        (x, y), (x - int(opticFlow[0][x, y]), y - int(opticFlow[1][x, y])))
                else:
                    normalizedVector = (
                        int(opticFlow[0][x, y]), int(opticFlow[1][x, y]))

                if ((x, y) != (x - normalizedVector[0], y + normalizedVector[1]) or showZeroVectors):

                    resultantMagnitude = calculateMagnitude(
                        (x, y), (x - int(opticFlow[0][x, y]), y - int(opticFlow[1][x, y])))
                    # print("drawing line at pos",(x,y), "resultant magnitude", resultantMagnitude)
                    totalMagnitude += resultantMagnitude

                    if (resultantMagnitude > magnitudeConstraint or resultantMagnitude == 0 and view):
                        mask = cv2.arrowedLine(mask, (x, y), (int(x + (normalizedVector[0]*vectorScaleFactor)), int(
                            y + (normalizedVector[1]*vectorScaleFactor)),), color[colorIterator].tolist(), 1, tipLength=.25)
            if (view):
                img = cv2.add(frameInitialResize, mask)
                img = cv2.resize(img, (1000, 1000))
        if (view):
            cv2.imshow('frame', img)
        if (writeFrames):
            filePath = "sampleFrames/frames" + str(int(isNormalizedVector)) + str(vectorScaleFactor) + str(filterType) + str(filterScaleX).replace(".", "-") + str(
                filterScaleY).replace(".", "-") + str(filterScaleT).replace(".", "-")+"FrameX-" + str(frameNumberX) + "FrameY" + str(frameNumberY) + str(magnitudeConstraint) + "/"
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            fileName = filePath + "frame" + str(frameNumber) + ".jpeg"
            if (view):
                cv2.imwrite(fileName, img)
        if (view):
            k = cv2.waitKey(frameTime)
            # reset frame
            mask = np.zeros_like(frameInitialResize)

        # break if escape key
        if (view):
            if k == 27:
                break
        if (frameNumber == frameToSample and frameToSample != 0):
            break
        frameNumber += 1
        colorIterator += 1
        frameInitialResize = frameDeltaResize
        frameInitial_blur = frameDelta_blur
        cornerList = gridView(frameDelta_blur, gridOffset)

    fileName = "dataFiles/AssayOutput" + str(int(isNormalizedVector)) + str(vectorScaleFactor) + str(filterType) + str(filterScaleX).replace(
        ".", "-") + str(filterScaleY).replace(".", "-") + str(filterScaleT).replace(".", "-") + str(magnitudeConstraint) + ".txt"
    with open(fileName, 'a+') as f:
        frameData = "frame row: " + str(frameNumberX) + "\nframe Column: " + str(frameNumberY) +\
            "\nAverage Magnitude per frame = " + str(totalMagnitude/frameNumber) + "\nframes sampled = " +\
            str(frameNumber) + "\n\n"
        f.write(frameData)

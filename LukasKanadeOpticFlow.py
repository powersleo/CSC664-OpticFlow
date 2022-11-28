
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
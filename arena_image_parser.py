import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def cropBattleInfo(img):
    # Return x,y coordinate of battle area
    _, threshold = cv.threshold(img, 230, 240, cv.THRESH_BINARY)
    threshold = 255 - threshold
    threshold = np.float32(threshold)

    dst = cv.cornerHarris(threshold, 20, 5, 0.04)
    borderMask = (dst > 0.5).astype(np.int32)
    borderCoord = np.where(borderMask == 1)

    minY = np.min(borderCoord[0])
    maxY = np.max(borderCoord[0])
    minX = np.min(borderCoord[1])
    maxX = np.max(borderCoord[1])

    return minX, maxX, minY, maxY

def charaRect(cropImg):
    # Find character square
    _, threshold = cv.threshold(cropImg, 230, 240, cv.THRESH_BINARY)
    threshold = np.float32(threshold)
    threshold = 255 - threshold

    edge = cv.Canny(cropImg, 170, 200)
    _, contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    allLineLength = []
    allCoord = []
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.07 * cv.arcLength(cnt, True), True)

        if len(approx) >= 4 and len(approx) < 6:
            # Length check
            lineLength = []
            for i in range(len(approx) - 1):
                dist = approx[i][0] - approx[i + 1][0]
                lineLength.append(np.sqrt(np.float(np.sum(dist ** 2))))

            # Angle check near to 90
            lineAngle = []
            for i in range(len(approx) - 2):
                vec1 = approx[i][0] - approx[i + 1][0]
                vec2 = approx[i + 1][0] - approx[i + 2][0]
                lineAngle.append(
                    np.abs(np.sum(vec1 * vec2) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))))
            # print (lineAngle)
            if np.min(lineLength) < 30 or np.mean(lineAngle) > 0.1 or np.var(lineLength) > 10:
                continue

            for i in range(len(approx) - 1):
                newline(approx[i][0], approx[i + 1][0])
            newline(approx[0][0], approx[-1][0])

            allCoord.append(approx)
            allLineLength.extend(lineLength)
            # print (lineLength)

    charWindow = np.ceil(np.median(allLineLength))

    xCoords = []
    yCoords = []
    for coords in allCoord:
        xCoords.extend(coords[:, 0, 0])
        yCoords.extend(coords[:, 0, 1])
    xCoords.sort()

    diff = []
    for idx in range(len(xCoords) - 1):
        if xCoords[idx + 1] - xCoords[idx] > charWindow * 0.8:
            diff.append(xCoords[idx])

    # Y Range of char img
    minY = np.min(yCoords)
    maxY = minY + int(charWindow)

    return charWindow, diff, minY, maxY

def starExtract(cropImg2):
    # Input is cropped char img
    star = cropImg2[int(cropImg2.shape[0] * 0.85):-2, 1:int(cropImg2.shape[1] * 0.7) + 1, :]
    newImg = (star[:, :, 1] > 180) * (star[:, :, 2] > 180) * (star[:, :, 0] < 180)

    starCnt = []
    starWidth = int(newImg.shape[1] / 5)
    for i in range(5):
        starCnt.append(np.mean(newImg[:, i * starWidth:(i + 1) * starWidth]))
    nStar = (np.sum(np.asarray(starCnt) > 0.2))

    return nStar

def newline(p1, p2): # For draw character area
    ax = plt.gca()

    xmin = np.min((p1[0], p2[0]))
    xmax = np.max((p1[0], p2[0]))
    ymin = np.min((p1[1], p2[1]))
    ymax = np.max((p1[1], p2[1]))

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='r')
    ax.add_line(l)
    return l

def featureExtract(cropImg2, kp2, des2):
    # cropImg2 : cropped character image
    # kp2, des2 : charlist image feature for matching

    # character recognition with SIFT feature
    matching = cv.cvtColor(cropImg2, cv.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(matching, None)
    matches = bf.knnMatch(des1, des2, k=4)

    # Apply ratio test
    good = []
    for a in matches:
        good.append(a)

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2.0)

    #im_out1 = cv.warpPerspective(matching, M, (db.shape[1], db.shape[0]))

    mid = np.expand_dims(np.array([cropImg2.shape[0] / 2, cropImg2.shape[1] / 2, 1.]), axis=0).transpose()
    loc = np.matmul(M, mid)

    return loc

# Open character list text
with open('char_list.txt', 'r') as f:
    charList = f.readlines()

for i in range(len(charList)):
    charList[i] = charList[i][:-1]
cCol = 16
#print (charList)

''' # Try to resnet feature, failed
# Network load
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# Feature load
charFeature = np.load('character.npy')
tfT = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
'''

# SIFT matching for db character list
db = cv.imread('characters.jpg',0) # character list img path
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp2, des2 = sift.detectAndCompute(db,None)

# BFMatcher with default params
bf = cv.BFMatcher(cv.NORM_L2)

# Test part for image recognition
name = 'test_img/3.png' # image file name
img = cv.imread(name, 0) # Process image for grayscale
colorImg = cv.imread(name) # Process image for color scale BGR
imgW, imgH = img.shape[::-1]

# Crop battle area
minX, maxX, minY, maxY = cropBattleInfo(img)

cropImg = img[minY:maxY, minX:maxX] # Battle information area with gray
cropColorImg = colorImg[minY:maxY, minX:maxX, :] # Battle information area with color

charWindow, diff, minY, maxY = charaRect(cropImg) # Char window size, diff is X coordnate (middle is VS), min/max Y

# Win/Lose check
#print (minY, diff[0], charWindow)
chk1 = cropColorImg[np.abs(minY-int(charWindow/2)):minY, diff[0]:int(charWindow)+diff[0], :]
winFlg = np.mean(chk1[:,:,0] < 80) > 0.1

# Parse First team
# Crop each chara img
for idx in range(5):
    plt.subplot(2,5,idx+1)
    cropImg2 = cropColorImg[minY:maxY, diff[idx]:diff[idx] + int(charWindow), :]

    loc = featureExtract(cropImg2, kp2, des2)

    # Fetch char name
    cIdx = int((cCol * int(loc[1] / 76) / 2 + int(loc[0] / 76) / 2))
    if cIdx < 0:
        cIdx = 0
    cName = charList[cIdx]

    nStar = starExtract(cropImg2)

    # Show plot
    plt.imshow(cv.cvtColor(cropImg2, cv.COLOR_BGR2RGB))

    if winFlg:
        plt.xlabel(cName + '\n' +  str(nStar)+', Win')
    else:
        plt.xlabel(cName + '\n' +  str(nStar) + ', Lose')

# Parse Second team
# Crop each chara img
for idx in range(5):
    plt.subplot(2,5,idx+6)
    cropImg2 = cropColorImg[minY:maxY, diff[idx+6]:diff[idx+6] + int(charWindow), :]

    loc = featureExtract(cropImg2, kp2, des2)

    cIdx = int((cCol * int(loc[1] / 76) / 2 + int(loc[0] / 76) / 2))
    if cIdx < 0:
        cIdx = 0
    cName = charList[cIdx]

    nStar = starExtract(cropImg2)

    plt.imshow(cv.cvtColor(cropImg2, cv.COLOR_BGR2RGB))

    if winFlg:
        plt.xlabel(cName + '\n' +  str(nStar) + ', Lose')
    else:
        plt.xlabel(cName + '\n' + str(nStar) + ', Win')

plt.show()
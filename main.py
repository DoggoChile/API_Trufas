import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def FLANN():
    root = os.path.abspath('..')
    img1path = os.path.join(root, 'API_Trufas Images', 'Luz1.jpg')
    img2path = os.path.join(root, 'API_Trufas Images', 'Luz2.jpg')
    img1 = cv.imread(img1path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2path, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    nKDTrees = 5
    nLeafChecks = 50
    nNeighbors = 2
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDTrees)
    searchParams = dict(checks=nLeafChecks)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(descriptor1, descriptor2, k=nNeighbors)
    matchesMask = [[0, 0] for i in range(len(matches))]
    testRatio = 0.01
    for i, (m, n) in enumerate(matches):
        if m.distance < testRatio * n.distance:
            matchesMask[i] = [1, 0]
    drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
    matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
    imgMatch = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    
    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

def FLANNHOMOGRAPHY():
    root = os.getcwd()
    img1path = os.path.join(root, 'Luz1.jpg')
    img2path = os.path.join(root, 'Luz2.jpg')
    img1 = cv.imread(img1path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2path, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    nKDTrees = 5
    nLeafChecks = 50
    nNeighbors = 2
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=nKDTrees)
    searchParams = dict(checks=nLeafChecks)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(descriptor1, descriptor2, k=nNeighbors)

    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    minGoodMatches = 20

    if len(goodMatches) > minGoodMatches:
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([keypoints2[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, errorThreshold)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        imgBorder = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        warpedImgBorder = cv.perspectiveTransform(imgBorder, M)
        img2 = cv.polylines(img2, [np.int32(warpedImgBorder)], True, 255, 3, cv.LINE_AA)
    else:
        print('Not enough matches')
        matchesMask = None

    green = (0, 255, 0)
    drawParams = dict(matchColor=green, singlePointColor=None, matchesMask=matchesMask, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    imgMatch = cv.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, None, **drawParams)


    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

FLANN()

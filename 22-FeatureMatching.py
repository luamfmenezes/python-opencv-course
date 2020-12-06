import numpy as np
import matplotlib.pyplot as plt
import cv2

# Brute-Force Matching with ORB Descriptors
# Brute-Force Matching with Sift Descriptors and Ratio Test
# FLANN based Matcher

reeses = cv2.imread('assets/images/reeses_puffs.png',0)

cereals = cv2.imread('assets/images/many_cereals.jpg',0)

# -------------------- ORB

orb = cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(reeses,mask=None)

kp2,des2 = orb.detectAndCompute(cereals,mask=None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

# matches[i].distance -> level of similarity

# Sort the matches by distance
matches = sorted(matches, key=lambda x:x.distance)

## Draw Lines where the matches are
reeses_matches = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:25], None)

# -------------------- Sift 

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(reeses,mask=None)
kp2,des2 = sift.detectAndCompute(cereals,mask=None)

bf = cv2.BFMatcher()

# k number of best matches
# Return one array of 2 matches [[match1,match2],...]
matches = bf.knnMatch(des1,des2,k=2)

goodMatches = []

## ratio test
## Less distance == better match
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        goodMatches.append([match1])

sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,goodMatches, None, flags=2)


# -------------------- FLANN based Matchers

# Much faster, but only find good matches but not necessarly the bests.

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(reeses,mask=None)
kp2,des2 = sift.detectAndCompute(cereals,mask=None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5) # {'algorithm'=FLANN_INDEX_KDTREE, 'threes'=5}
search_params = dict(checks=50) # {'cheks':50}

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

## ratio test
## Less distance == better match
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.75*match2.distance:
        ## [1,0] to select only the best match, is the same that .append([match1])
        matchesMask[i] = [1,0]

draw_params = dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0)

flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches, None, **draw_params)

plt.imshow(flann_matches)

plt.show()


import cv2 as cv
import cv2
from cv2 import imread
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

base_path = '/Users/apple/Downloads'
images = os.listdir(join(base_path, 'images_blocks'))
img = imread(join(base_path, 'images_blocks', images[1]), 0)
ref_img = imread(join(base_path, 'basic_image_blocks.png'), 0)

MAX_FEATURES = 50
GOOD_MATCH_PERCENT = 0.5


def alignImages(im1, im2):
    orb = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)

    keypts1, desc1 = orb.detectAndCompute(im1, None)
    keypts2, desc2 = orb.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches = matcher.match(desc1, desc2)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    imMatches = cv2.drawMatches(im1, keypts1, im2, keypts2, matches, None)
    cv2.imwrite('matches.jpg', imMatches)

    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts1[i, :] = keypts1[match.queryIdx].pt
        pts2[i, :] = keypts2[match.queryIdx].pt
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


imReg, h = alignImages(img, ref_img)

# plt.subplot(221), plt.title('Reference Image')
# imshow(ref_img, cmap='gray')
# plt.subplot(222), plt.title('Registring Image')
# imshow(img, cmap='gray')
# plt.subplot(212), plt.title('Registered image')
#
# imshow(imReg, cmap='gray')
plt.imsave('output1.png', imReg, cmap='gray')
# -*- coding: utf-8 -*-
# @Time : 2022/4/10 18:44
# @Author : zzy
# @File : panoramic_stitching
# @Description : 

import numpy as np
import cv2


class panoramic_stitcher:
    def __init__(self, ratio=0.75, reprojThresh=4.0):
        self.ratio = ratio
        self.reprojThresh = reprojThresh

    def stitch(self, img1, img2, save_path):
        '''
        External interface, realize panoramic stitching
        :param img1: image1
        :param img2: image2
        :param save_path: Result Saving path
        :return: stitching result
        '''

        while img1.shape[0] > 800:
            img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
            img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)

        keypoints1, descriptors1 = self.__get_keypoints_and_descriptors(img1)
        keypoints2, descriptors2 = self.__get_keypoints_and_descriptors(img2)

        # calculate the feature point matching and transformation matrix H
        M = self.__match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2)

        # If the returned result is empty and no feature points are successfully matched, the algorithm exits
        if M is None:
            return None

        matches, H, status = M
        # Img1 is applied to the transformation matrix H
        result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        # Pass picture B to the leftmost end of the result picture
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
        cv2.imwrite(save_path, result)
        return result

    def __get_keypoints_and_descriptors(self, img):
        '''
        get sift keypoints and descriptors
        :param image: input images
        :return: location of keypoints, descriptors
        '''
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # make sift descriptor
        descriptor = cv2.SIFT_create()
        # detect sift keypoints and compute descriptors
        keypoints, descriptors = descriptor.detectAndCompute(img, None)
        # get location of key points
        keypoints_loc = np.float32([keypoint.pt for keypoint in keypoints])
        return keypoints_loc, descriptors

    def __match_keypoints(self, keypoints1, keypoints2, descriptors1, descriptors2):
        '''
        calculate the feature point matching and transformation matrix H
        :param keypoints1:
        :param keypoints2:
        :param descriptors1:
        :param descriptors2:
        :return: Matching results, transformation matrix
        '''
        # Brute Force match
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        matches = []
        for m in raw_matches:
            # When the ratio of the nearest distance to the next closest distance is less than the ratio value,
            # the matching pair is retained
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # When the matching pairs after screening are greater than 4,
        # the perspective transformation matrix is calculated
        if len(matches) > 4:
            ptsA = np.float32([keypoints1[i] for (_, i) in matches])
            ptsB = np.float32([keypoints2[i] for (i, _) in matches])
            # The Angle change matrix is calculated
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)
            return matches, H, status
        return None


if __name__ == '__main__':
    img_num = 8
    imgs = []
    for i in range(img_num):
        imgs.append(cv2.imread(f'img{i + 1}.jpg'))

    # 把图片拼接成全景图
    stitcher = panoramic_stitcher()

    for i in range(int(img_num / 2)):
        stitcher.stitch(imgs[i * 2], imgs[i * 2 + 1], f'stitching_result_{i * 2 + 1}_{i * 2 + 2}.jpg')

# -*- coding: utf-8 -*-
# @Time : 2022/4/9 18:46
# @Author : zzy
# @File : dog_dector
# @Description : 

import cv2
import numpy as np
from functools import cmp_to_key


class dog_keypoint_detector():
    def __init__(self):
        self.sigma = 1.6
        self.assumed_blur = 0.5
        self.interval_num = 3
        self.contrast_threshold = 0.04
        self.border_width = 5
        self.eigenvalue_ratio = 10
        self.num_attempts_until_convergence = 5

    def get_key_points(self, img):
        '''
        get key points
        :param image: imput image
        :return: list of key points
        '''
        self.__get_first_image(img)  # get first image of the first octave
        self.__get_sigma_of_gaussian()  # get sigma of all images in all octaves
        self.__get_octave_pyramid()  # get octave pyramid
        self.__get_dog_imgs()  # get diffience of gaussian images
        self.__get_extrema()  # get extrema of dog
        self.__drop_duplicates()  # drop duplicated pixels
        self.__convert_to_input_size()  # convert form the first image to original image
        return self.keypoints

    def __get_first_image(self, img):
        '''
        get first image of the first octave
        :param img: imput image
        '''
        img = img.astype('float32')
        img = cv2.resize(img, None, fx=2, fy=2)
        sigma_diff = np.sqrt(max((self.sigma ** 2) - ((2 * self.assumed_blur) ** 2), 0.01))
        img = cv2.GaussianBlur(img, None, sigma_diff)
        self.base_img = img

    def __get_sigma_of_gaussian(self):
        '''
        get sigma of all images in all octaves
        :return:
        '''
        self.octave_num = int(round(np.log(min(self.base_img.shape)) / np.log(2) - 1))
        img_num_in_one_oct = self.interval_num + 3
        k = 2 ** (1.0 / self.interval_num)
        gaussian_kernels = [self.sigma]
        for i in range(1, img_num_in_one_oct):
            gaussian_kernels.append(self.sigma * np.sqrt((k ** i) ** 2 - (k ** (i - 1)) ** 2))
        self.gaussion_kernels = gaussian_kernels

    def __get_octave_pyramid(self):
        '''
        get octave pyramid, do gaussian blurs
        '''
        gaussian_imgs = []
        img = self.base_img.copy()

        for i in range(self.octave_num):
            gaussian_imgs_in_octave = []
            gaussian_imgs_in_octave.append(img)
            for gaussian_kernel in self.gaussion_kernels[1:]:
                img = cv2.GaussianBlur(img, None, gaussian_kernel)
                gaussian_imgs_in_octave.append(img)
            gaussian_imgs.append(gaussian_imgs_in_octave)
            octave_base = gaussian_imgs_in_octave[-3]
            img = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)))
        self.gaussion_images = np.array(gaussian_imgs, dtype=object)

    def __get_dog_imgs(self):
        '''
        compute different of gaussian
        '''
        dog_images = []

        for gaussian_imgs_in_octave in self.gaussion_images:
            dog_imgs_in_octave = []
            for first_image, second_image in zip(gaussian_imgs_in_octave, gaussian_imgs_in_octave[1:]):
                dog_imgs_in_octave.append(cv2.subtract(second_image,
                                                       first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_imgs_in_octave)
        self.dog_imgs = np.array(dog_images, dtype=object)
        del self.gaussion_images

    def __get_extrema(self):
        '''
        get extrema of dog images
        '''
        threshold = np.floor(0.5 * self.contrast_threshold / self.interval_num * 255)  # from OpenCV implementation
        self.keypoints = []
        for oct_i, dog_imgs_in_oct in enumerate(self.dog_imgs):
            for img_i, (first_img, second_img, third_img) in enumerate(
                    zip(dog_imgs_in_oct, dog_imgs_in_oct[1:], dog_imgs_in_oct[2:])):
                # (i, j) is the center of the 3x3 array
                for i in range(self.border_width, first_img.shape[0] - self.border_width):
                    for j in range(self.border_width, first_img.shape[1] - self.border_width):
                        if self.__is_pixel_extrema(first_img[i - 1:i + 2, j - 1:j + 2],
                                                   second_img[i - 1:i + 2, j - 1:j + 2],
                                                   third_img[i - 1:i + 2, j - 1:j + 2], threshold):
                            keypoint_with_img_i = self.__fit_extrema_pos(i, j, img_i + 1, oct_i, dog_imgs_in_oct)
                            if keypoint_with_img_i is not None:
                                self.keypoints.append(keypoint_with_img_i[0])

    def __is_pixel_extrema(self, first_layer, second_layer, third_layer, threshold):
        '''
        Calculates whether a pixel is larger than 26 surrounding pixels
        :param first_layer: first layer
        :param second_layer: second layer
        :param third_layer: third layer
        :param threshold: the minimum of the value of the pixel
        :return: whether the pixel is an extrema
        '''
        center_pixel_value = second_layer[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_layer) and \
                       np.all(center_pixel_value >= third_layer) and \
                       np.all(center_pixel_value >= second_layer[0, :]) and \
                       np.all(center_pixel_value >= second_layer[2, :]) and \
                       center_pixel_value >= second_layer[1, 0] and \
                       center_pixel_value >= second_layer[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_layer) and \
                       np.all(center_pixel_value <= third_layer) and \
                       np.all(center_pixel_value <= second_layer[0, :]) and \
                       np.all(center_pixel_value <= second_layer[2, :]) and \
                       center_pixel_value <= second_layer[1, 0] and \
                       center_pixel_value <= second_layer[1, 2]
        return False

    def __fit_extrema_pos(self, i, j, img_i, oct_i, dog_imgs_in_oct):
        '''
        Iteratively fit the location of the extreme point by the second order Taylor expansion
        :param i: index of the extreme
        :param j: index of the extreme
        :param img_i: index of image
        :param oct_i: index of octave
        :param dog_imgs_in_oct: dog images in the octave
        :return: key points, image index
        '''
        extremum_is_outside_img = False
        img_shape = dog_imgs_in_oct[0].shape
        for attempt_index in range(self.num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            first_img, second_img, third_img = dog_imgs_in_oct[img_i - 1:img_i + 2]
            pixel_cube = np.stack([first_img[i - 1:i + 2, j - 1:j + 2],
                                   second_img[i - 1:i + 2, j - 1:j + 2],
                                   third_img[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
            gradient = self.__compute_gradient(pixel_cube)
            hessian = self.__compute_hessian(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            img_i += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the img
            if i < self.border_width or i >= img_shape[0] - self.border_width or j < self.border_width or j >= \
                    img_shape[
                        1] - self.border_width or img_i < 1 or img_i > self.interval_num:
                extremum_is_outside_img = True
                break
        if extremum_is_outside_img:
            return None
        if attempt_index >= self.num_attempts_until_convergence - 1:
            return None
        value_at_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(value_at_extremum) * self.interval_num >= self.contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and self.eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                    (self.eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = cv2.KeyPoint()
                keypoint.pt = (
                    (j + extremum_update[0]) * (2 ** oct_i), (i + extremum_update[1]) * (2 ** oct_i))
                keypoint.octave = oct_i + img_i * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (
                        2 ** 16)
                keypoint.size = self.sigma * (2 ** ((img_i + extremum_update[2]) / np.float32(self.interval_num))) * (
                        2 ** (oct_i + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(value_at_extremum)
                return keypoint, img_i
        return None

    def __compute_gradient(self, pixel_array):
        '''
        compute the gradient using discrete value
        :param pixel_array: 3*3*3 value
        '''
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def __compute_hessian(self, pixel_array):
        '''
        compute hessian using discrete value
        :param pixel_array: 3*3*3 value
        :return:
        '''
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]])

    def __drop_duplicates(self):
        '''
        drop duplicated points
        '''
        if len(self.keypoints) < 2:
            return self.keypoints

        self.keypoints.sort(key=cmp_to_key(self.__compare_keypoints))
        unique_keypoints = [self.keypoints[0]]

        for next_keypoint in self.keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                    last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                    last_unique_keypoint.size != next_keypoint.size or \
                    last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        self.keypoints = unique_keypoints

    def __compare_keypoints(self, keypoint1, keypoint2):
        '''
        define the comparison of two points
        :param keypoint1: key point 1
        :param keypoint2: key point 2
        '''
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def __convert_to_input_size(self):
        '''
        convert form the first image to original image
        '''
        converted_keypoints = []
        for keypoint in self.keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        self.keypoints = converted_keypoints


if __name__ == '__main__':
    pathes = ['butterfly.png', 'far1.jpg', 'near1.jpg']
    detector = dog_keypoint_detector()
    for path in pathes:
        image_color = cv2.imread(path)
        image = cv2.imread(path, 0)
        keypoints = detector.get_key_points(image)
        for point in keypoints:
            cv2.circle(image_color, (int(point.pt[0]), int(point.pt[1])), int(point.size), (0, 0, 255))
        cv2.imwrite(path[:-4] + '_with_point.png', image_color)

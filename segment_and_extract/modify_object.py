import cv2
from dataclasses import dataclass
import numpy as np
import typing

from segment_and_extract.utils import draw_point_on_plot, find_hand_center

@dataclass
class ModifyObject(object):
    image: np.ndarray
    thermal_image: np.ndarray
    threshold_value: int
    debug_mode: bool = False

    def __del__(self):
        cv2.destroyAllWindows()

    def __post_init__(self):
        angle = self.need_rotate_image(self.thermal_image)
        self.modified_image = self.rotate_image(self.image, angle)
        self.modified_thermal_image = self.rotate_image(self.thermal_image, angle)
        # self.crop_images()

    def need_rotate_image(self, thermal_image):
        y, x = thermal_image.shape

        image_frame_avg_tmp = {0: np.average(thermal_image[y - 1, :]),  # bottom - dont rotate
                               270: np.average(thermal_image[:, x - 1]),  # right - rotate by 270
                               180: np.average(thermal_image[0, :]),  # upper - rotate by 180
                               90: np.average(thermal_image[:, 0])}  # left - rotate by 90

        rotate_by = sorted(image_frame_avg_tmp.items(), key=lambda x: x[1], reverse=True)

        # return rotate_by[0][0]
        return 0

    def rotate_image(self, mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        if angle == 0:
            return mat.copy()

        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def crop_images(self):
        y, x, c = self.modified_image.shape
        frame = self.modified_image.copy()
        cnts = self.find_hand_contours(frame)
        centerMass = find_hand_center(cnts)

        if self.debug_mode:
            print(y, x, c)
            draw_point_on_plot(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR), (int(x/2),int(y/2)), color=(255,255,0))
            draw_point_on_plot(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),centerMass)

        factor = 1
        while centerMass[1] - int(y/2) >= 80 and factor > 0 :

            y, x, c = frame.shape
            draw_point_on_plot(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR), (int(x / 2), int(y / 2)), color=(255, 255, 0))
            frame = self.crop_all_images([0,y/2], frame, factor)
            factor += 1
            image = frame.copy()
            cnts = self.find_hand_contours(image)
            centerMass = find_hand_center(cnts)
            if self.debug_mode:
                print(y, x, c)
                draw_point_on_plot(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),centerMass)
            y, x, c = frame.shape

        self.modified_image = frame[0: y, int(x / 2 - 200):  int((x / 2) + 220)]
        self.modified_thermal_image = self.modified_thermal_image[0: y, int(x / 2 - 210):  int((x / 2) + 210)]


    def crop_all_images(self, centerMass, frame, x):
        # factor = 0.14 * x
        factor = 0.9 * x

        self.modified_thermal_image = self.modified_thermal_image[85:  int(centerMass[1] + factor * centerMass[1]), : ]
        return frame[85:  int(centerMass[1] + factor * centerMass[1]) , :]

    def find_hand_contours(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', img_gray)
        blur = cv2.blur(img_gray, (3, 3))
        # cv2.imshow('blur', blur)
        # ret, mask2 = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
        ret, mask2 = cv2.threshold(img_gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        cnts = self.find_contours(frame, mask2)
        return cnts

    def find_contours(self, frame, mask2):
        # Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw Contours
        # cv2.drawContours(frame, contours, -1, (122,122,0), 3)
        # cv2.drawContours(frame, contours, -1, (0, 0, 0), 3)
        # cv2.imshow('contours', frame)
        # cv2.imshow('Dilation',median)
        # Find Max contour area (Assume that hand is in the frame)
        max_area = 100
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i

            # Largest area contour
        cnts = contours[ci]
        return cnts
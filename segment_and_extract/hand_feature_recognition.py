import os

import time
from typing import List

from segment_and_extract.utils import *
from old_versions.calculate_attributes import CalculateAttribute

import resources as res


# # Open Camera object
# cap = cv2.VideoCapture(0)
#
# # Decrease frame size
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)
#

def nothing(x):
    pass


class HandFeatureRecognition():

    def __init__(self, image,threshold_value, debug_mode=False):
        self.original_image = image
        self._image = image.copy()
        self.feature_coordinates = {}
        self.extra_points = {}
        self.center = ""
        self.end_process = None
        self.threshold_value=threshold_value
        self.debug_mode = debug_mode

    def __del__(self):
        cv2.destroyAllWindows()

    def find_feature_coordinates(self):
        cv2.imshow('raw_image', self._image)

        frame = self._image
        mask2 = self.filter_out_backgroud_noise(frame)

        cnts = self.find_contours(frame, mask2)

        centerMass = find_hand_center(cnts)
        self.center = centerMass

        # Draw center mass
        cv2.circle(frame, centerMass, 7, [100, 0, 255], -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Center', tuple(centerMass), font, 0.5, (255, 255, 255), 2)

        # Find convex hull
        hull = cv2.convexHull(cnts,clockwise=True, returnPoints = False)
        # Find convex defects
        # hull2 = cv2.convexHull(cnts, returnPoints=False)
        defects = cv2.convexityDefects(cnts, hull)

        # Get defect points and draw them in the original image
        l_startend = []
        # tmp_fat = tuple(cnts[defects[0, 0][2]][0])
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])  # the farthest defect point between two hull  s and e with distance of f
            # distance_in_between = FindDistance(far, tmp_fat)
            tmp_fat = far
            # if (distance_in_between == 0 or distance_in_between >50) :
            if d > 2000 :
                self.is_point_too_close(l_startend, start, end, frame)
                # FarDefect.append(far)
                self.feature_coordinates[len(self.feature_coordinates)] = far
                cv2.line(frame, start, end, [229,  255, 204], 2)
                cv2.circle(frame, far, 10, [100, 100, 255], 3)

        # # **********
        cv2.imshow('FarDefect', frame)
        # # **********


        self.end_process = frame
        self.sort_point_by_angle()

        return mask2

    def is_point_too_close(self, l_startend , start, end, frame):

        close = [0 ,0 ]
        for point in l_startend:
            # print(FindDistance(start, point))
            # print(FindDistance(end, point))
            if FindDistance(start, point) < 40 :
                close[0] = 1
            if FindDistance(end, point) < 40 :
                close[1] = 1

        if close[0] == 0 and self.center[1] > start[1] :
            l_startend.append(start)
            self.feature_coordinates[len(self.feature_coordinates)] = start
            cv2.circle(frame, start, 10, [100, 100, 255], 3)

        if close[1] == 0 and self.center[1] > end[1] + 110:
            l_startend.append(end)
            self.feature_coordinates[len(self.feature_coordinates)] = end
            cv2.circle(frame, end, 10, [100, 100, 255], 3)

    def find_palm_coordinates(self, list_coordinates: List[tuple]) -> None:
        if not list_coordinates:
            return None
        img = self.end_process.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # find mid point fo the wrist
        wrist_lower_start_point = list_coordinates[0]
        wrist_lower_end_point = list_coordinates[-1]
        self.extra_points['mid_point_wrist'] = self.midpoint(wrist_lower_start_point, wrist_lower_end_point)
        cv2.circle(self.end_process, self.extra_points['mid_point_wrist'], 10, [100, 100, 255], 3)
        if self.debug_mode:
            cv2.line(img, wrist_lower_start_point, wrist_lower_end_point, [0, 255, 255], 3, cv2.LINE_AA)
            cv2.circle(img, wrist_lower_start_point, 5, (255, 255, 0), -1)
            cv2.circle(img, wrist_lower_end_point, 5, (255, 255, 0), -1)
            cv2.circle(img, self.extra_points['mid_point_wrist'], 5, (0, 0, 205), -1)

        # find mid point of thumb blob
        space_before_thumb = list_coordinates[-10]
        self.extra_points['mid_point_thumb_blob'] = self.midpoint(space_before_thumb, self.extra_points['mid_point_wrist'])
        cv2.circle(self.end_process, self.extra_points['mid_point_thumb_blob'], 10, [100, 100, 255], 3)

        if self.debug_mode:
            cv2.line(img, space_before_thumb, self.extra_points['mid_point_wrist'], [0, 255, 255], 3, cv2.LINE_AA)
            cv2.circle(img, space_before_thumb, 5, (255, 255, 0), -1)
            cv2.circle(img, self.extra_points['mid_point_thumb_blob'], 5, (0, 0, 205), -1)

        # find mid point after pinky
        space_after_pinky = list_coordinates[-2]
        self.extra_points['mid_point_edge_blob'] = self.midpoint(space_after_pinky, self.extra_points['mid_point_wrist'])
        cv2.circle(self.end_process, self.extra_points['mid_point_edge_blob'], 10, [100, 100, 255], 3)


        if self.debug_mode:
            cv2.line(img, space_after_pinky, self.extra_points['mid_point_wrist'], [0, 255, 255], 3, cv2.LINE_AA)
            cv2.circle(img, space_after_pinky, 5, (255, 255, 0), -1)
            cv2.circle(img, self.extra_points['mid_point_edge_blob'], 5, (0, 0, 205), -1)

            plt.imshow(img)
            plt.show()

    def midpoint(self, p1, p2):
        return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

    def sort_point_by_angle(self):
        img = self.end_process.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sorted_y_coordinates = sorted(list(self.feature_coordinates.values()), key=lambda x: x[1])
        wrist_lower_p = sorted_y_coordinates[-1]
        if self.debug_mode:
            cv2.circle(img, wrist_lower_p, 5, (255, 255, 255), -1)
            cv2.line(img, wrist_lower_p, self.center, [102, 0, 51], 3, cv2.LINE_AA)

        coordinates_by_angle = {}

        for _, point in self.feature_coordinates.items():
            angle = calculate_angle_by_three_points(wrist_lower_p, point, self.center)
            if angle in coordinates_by_angle.keys():
                angle += 1
            coordinates_by_angle[angle] = point
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.debug_mode:
                cv2.line(img, self.center, point, [102, 0, 51], 3, cv2.LINE_AA)
                cv2.putText(img, str(angle), (point[0] - 10, point[1] - 10), font, 0.8, (255, 255, 255),2)
                cv2.circle(img, point, 5, (255, 255, 255), -1)

        self.feature_coordinates = coordinates_by_angle
        if self.debug_mode:
            plt.imshow(img)
            plt.show()

        # list sort not clockwise point when wrist_lower is the first
        sorted_coordinates_not_clockwise = [coordinates_by_angle[key] for key in sorted(coordinates_by_angle.keys())]

        # check if right hand or reversed palm and reverse for the coordinate order
        if self.center[0] > wrist_lower_p[0]:
            sorted_coordinates_not_clockwise.reverse()
        return sorted_coordinates_not_clockwise


    def find_contours(self, frame, mask2):
        # Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw Contours
        cv2.drawContours(frame, contours, -1, (122,122,0), 3)

        cv2.drawContours(frame, contours, -1, (0, 0, 0), 3)
        cv2.imshow('contours', frame)
        cv2.imshow('Dilation',median)
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

    def filter_out_backgroud_noise(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', img_gray)
        blur = cv2.blur(img_gray, (3, 3))
        cv2.imshow('blur', blur)
        ret, mask2 = cv2.threshold(img_gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imshow('mask2', mask2)

        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(mask2, kernel_square, iterations=1)
        dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        cv2.imshow('erosion', dilation2)
        filtered = cv2.medianBlur(dilation2, 5)
        cv2.imshow('filtered', filtered)

        ret, thresh = cv2.threshold(filtered, 127, 255, 0)
        cv2.imshow('thresh2', thresh)
        # return mask2
        return thresh


    def tuning_method(self):
        cv2.namedWindow('HSV_TrackBar')
        cv2.namedWindow('img')
        # Creating track bar
        cv2.createTrackbar('h', 'HSV_TrackBar', 0, 179, nothing)
        cv2.createTrackbar('s', 'HSV_TrackBar', 0, 255, nothing)
        cv2.createTrackbar('v', 'HSV_TrackBar', 0, 255, nothing)

        while (1):
            cv2.setMouseCallback('img', click_event, param={'frame': frame})
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Capture frames from the camera
            frame = self._image

            # Blur the image
            blur = cv2.blur(frame, (3, 3))

            # Convert to HSV color space
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # Create a binary image with where white will be skin colors and rest is black
            mask2 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 248, 255]))

from math import atan2, degrees

import cv2
import numpy as np
from matplotlib import pyplot as plt

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x,'  ', y)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(param['frame'] , f' x: {x} y: {y}', (x, y), font, 0.5, (255, 255, 0), 2)
        cv2.imshow('img_window', param['frame'])


# Function to find angle between two vectors
def Angle(v1, v2):
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def calculate_angle_by_three_points(p1, p2, center):
    p1c = np.subtract(p1, center)
    p2c = np.subtract(p2 , center)
    ang_a = np.arctan2(*p1c[::-1])
    ang_b = np.arctan2(*p2c[::-1])
    return int(np.rad2deg((ang_a - ang_b) % (2 * np.pi)))


# Function to find distance between two points in a list of lists
def FindDistance(A, B):
    return np.sqrt(np.power((A[0] - B[0]), 2) + np.power((A[1] - B[1]), 2))


def draw_point_on_plot(image, point, color=(255, 255, 255)):
    cv2.circle(image, point, 5, color, -1)
    plt.imshow(image)
    plt.show()

def find_hand_center(cnts):
    # Find moments of the largest contour
    moments = cv2.moments(cnts)
    # Central mass of first order moments
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)
    return centerMass


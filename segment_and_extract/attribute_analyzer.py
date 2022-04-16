import os
import statistics

import numpy as np
import flirimageextractor
import pandas as pd
from matplotlib import cm

from segment_and_extract.implicit_features_extractor import adding_fingertips_statistics, manage_add_implicit_data
from segment_and_extract.utils import *
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew

from metrics_utils.feature_metrics import FeatureMetrics
from resources import output_folder, ROOT_DIR, metric_csv_files


class AttributeAnalyzer():

    def __init__(self , image, thermal_img, hand_mask,debug_mode=False):
        self.original_image = image
        self.thermal_image = thermal_img
        self.debug_mode = debug_mode
        self.dst_img_weighted = image.copy()
        self.hand_mask = hand_mask
        self.metric_list = {'mean': self.calculate_mean,'kurtosis': self.calculate_kurtosis,'skewness': self.calculate_skewness,
                            'median': self.calculate_median, 'sd' :self.calculate_standard_deviation,
                            'variance': self.calculate_variance, 'min_tempe': self.calculate_min_temperature,
                            'max_tempe' :self.calculate_max_temperature}
        self.fm = FeatureMetrics()



    def create_matrix_attributes(self,point_coordinate):

        for angle, point in point_coordinate.items():
            tempe_point_values = self.collect_values(point)
            self.run_all_metrices(angle, tempe_point_values)
            # self.fm.add_metric(str(angle), 'location', str(point))

        if self.debug_mode:
            plt.imshow(self.dst_img_weighted)
        plt.show()

    def add_implicit_points(self, file_name):
        ###############
        metric_feature_output = os.path.join(ROOT_DIR, metric_csv_files, 'built_features_'+file_name+ '_0.csv')

        metrics = pd.DataFrame(columns=['location'])
        metrics.set_index('location', inplace=True)
        implicit_data = manage_add_implicit_data(self.fm.metrics)

        for loc_name, values in implicit_data.items():
            for metric_name, metric_function in self.metric_list.items():
                metrics.loc[loc_name, metric_name] = metric_function(pd.to_numeric(values))

        metrics.to_csv(metric_feature_output)

        ########### old version ####################
        # implicit_data = manage_add_implicit_data(self.fm.metrics)
        # for loc_name, values in implicit_data.items():
        #     self.run_all_metrices(loc_name, pd.to_numeric(values))

        ###############


    def collect_values(self,center, radius=30):
        y, x = self.thermal_image.shape
        mask = np.zeros(self.thermal_image.shape, np.uint8)
        cv2.circle(mask, (center[0],center[1]),radius, (255,255,255) ,-1)
        mask = cv2.bitwise_and(mask,self.hand_mask)
        y , x = np.where(mask==255)

        intensity_values_from_original = self.thermal_image[y, x]

        # cv2.imshow('point', intensity_values_from_original)

        # intensity_values_from_original = [ temp for temp in intensity_values_from_original if temp > MIN_TEMP]
        if self.debug_mode:
            self.dst_img_weighted = cv2.addWeighted(self.thermal_image, 0.7, mask, 0.3, 0, dtype = cv2.CV_8U)
            plt.imshow(self.dst_img_weighted)
            plt.show()



        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(self.original_image, average_temp, (center[0],center[1]), font, 1, (255, 255, 255), 1)

        return intensity_values_from_original

    def run_all_metrices(self,angle,  values):

        for metric_name, metric_function in self.metric_list.items():
            self.fm.add_metric(str(angle), metric_name, metric_function(values))


    def calculate_mean(self, list_of_temp):
        return str(round(statistics.mean(list_of_temp), 2))


    def calculate_kurtosis(self, list_of_temp):
        return str(round(kurtosis(list_of_temp), 4))

    def calculate_skewness(self, list_of_temp):
        # skewness = 0 : normally distributed.
        # skewness > 0 : more weight in the left tail of the distribution.
        # skewness < 0 : more weight in the right tail of the distribution.

        return str(round(float(skew(list_of_temp)),2))

    def calculate_median(self, list_of_temp):
        return str(round(statistics.median(list_of_temp), 2))

    def calculate_standard_deviation(self, list_of_temp):
        return str(round(statistics.stdev(list_of_temp), 4))

    def calculate_variance(self, list_of_temp):
        return str(round(statistics.variance(list_of_temp), 4))

    def calculate_min_temperature(self, list_of_temp):
        return str(round(min(list_of_temp), 2))

    def calculate_max_temperature(self, list_of_temp):
        return str(round(max(list_of_temp), 2))





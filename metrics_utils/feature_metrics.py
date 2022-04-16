import numpy
import pandas as pd
from singleton_decorator import singleton

@singleton
class FeatureMetrics:
    ANGLE_COL = "Angle"
    NONE_VALUE = numpy.nan
    metrics = pd.DataFrame(columns=[ANGLE_COL])

    def __init__(self):
        self.metrics.set_index(self.ANGLE_COL, inplace=True)

    def init_metric(self):
        self.metrics = pd.DataFrame(columns=[self.ANGLE_COL])
        self.metrics.set_index(self.ANGLE_COL, inplace=True)

    def add_metric(self, angle,  metric_name, value):
        self.metrics.loc[angle, metric_name] = value


    def sort_by_index(self):
        str_index = {'center': -1, 'mid_point_edge_blob': -2, 'mid_point_thumb_blob': -3, 'mid_point_wrist': -4,
                     'fingertips': -5, 'inbetween':-6, 'all_palm_center': -8}

        for index_val in self.metrics.index:
            if index_val in str_index.keys():
                self.metrics.loc[index_val, 'num_index'] = str_index[index_val]
            else:
                self.metrics.loc[index_val, 'num_index'] = int(index_val)

        self.metrics = self.metrics.sort_values(['num_index']).drop('num_index', axis=1)
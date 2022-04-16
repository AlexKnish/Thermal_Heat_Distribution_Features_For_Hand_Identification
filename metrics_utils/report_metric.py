import numpy
import pandas as pd
from singleton_decorator import singleton

@singleton
class ReportMetric:
    FILE_COL = "file_name"
    NONE_VALUE = numpy.nan
    metrics = pd.DataFrame(columns=[FILE_COL])

    def __init__(self):
        self.metrics.set_index(self.FILE_COL, inplace=True)


    def add_metric(self, file_name,  metric_name, value):
        self.metrics.loc[file_name, metric_name] = value



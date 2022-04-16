import os

from data_preparation.matrix_manipulation import convert_matrix_to_single_row, convert_matrix_to_average_row
from data_preparation.preprocessing_data import mean_removal_and_variance_scaling, scaling_feature_in_range, \
    normalization

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(ROOT_DIR, "thermal_images")
raw_images = os.path.join(ROOT_DIR, "01_process_raw_images")
# raw_images = os.path.join(ROOT_DIR, "data_acquisition")
extracted_images = os.path.join(ROOT_DIR, "02_process_extracted_thermal_images")
output_folder = os.path.join(ROOT_DIR, "03_process_thermal_outputs")
reports = os.path.join(ROOT_DIR, "04_reports")
metric_csv_files = os.path.join(ROOT_DIR, "05_metric_csv_files")
final_matrices = os.path.join(ROOT_DIR, "06_final_matrices")
model_presentation = os.path.join(ROOT_DIR, "07_excel_model_presentation")


matrix_manipulation = {'single_row' : convert_matrix_to_single_row,
                       'matrix_to_average_row' : convert_matrix_to_average_row}

data_preparation = dict(z_score=mean_removal_and_variance_scaling, min_max_scale=scaling_feature_in_range,
                        normalization=normalization)
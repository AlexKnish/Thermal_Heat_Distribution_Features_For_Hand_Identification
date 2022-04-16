import glob
import pathlib
from shutil import copy

from segment_and_extract.flir_extractor_util import FlirExtractorUtil
from segment_and_extract.hand_feature_recognition import HandFeatureRecognition
from segment_and_extract.attribute_analyzer import AttributeAnalyzer
import os
import cv2

from segment_and_extract.modify_object import ModifyObject
from metrics_utils.feature_metrics import FeatureMetrics
from metrics_utils.report_metric import ReportMetric
from resources import ROOT_DIR, data_folder, raw_images, extracted_images, output_folder, metric_csv_files


def main(debug_mode):
    feature_metrics = FeatureMetrics()
    rm = ReportMetric()
    thermal_files_list = glob.glob(os.path.join(raw_images,"*.jpg"))
    features = {}
    for thermal_image_name in thermal_files_list:
        thermal_image_name = os.path.basename(thermal_image_name)
        print(thermal_image_name)

        copy(os.path.join(raw_images,thermal_image_name), os.path.join(extracted_images))
        file_name, file_type = thermal_image_name.split('.')
        #
        flir = FlirExtractorUtil(extracted_images)
        #
        # a = flir.get_thermal_image(thermal_image_name)

        flir.extract_thermal_img_from_raw_img(thermal_image_name)

        thermal_type = get_thermal_file_type(color_map=3)
        image_path = os.path.join(extracted_images, file_name + thermal_type + file_type)
        image = cv2.imread(image_path)

        thermal_image = flir.get_thermal_image(thermal_image_name)
        threshold_value = calculate_threshold_val(thermal_image)
        md = ModifyObject(image, thermal_image,threshold_value, debug_mode)

        hr = HandFeatureRecognition(md.modified_image,threshold_value, debug_mode=debug_mode)
        hand_mask = hr.find_feature_coordinates()
        sorted_anticlockwise = hr.sort_point_by_angle()
        hr.find_palm_coordinates(sorted_anticlockwise)
        all_points = gather_all_points(hr)
        print(len(all_points))

        aa = AttributeAnalyzer(md.modified_image, md.modified_thermal_image, hand_mask, debug_mode=True)
        aa.create_matrix_attributes(all_points)

        feature_metrics.sort_by_index()
        aa.add_implicit_points(file_name)
        feature_metrics.sort_by_index()


        # rm.metrics.to_csv(os.path.join(ROOT_DIR, '04_reports', 'metric_report.csv'))
        rm.metrics.to_csv(os.path.join(ROOT_DIR, '04_reports', 'metric_report.csv'))
        metric_feature_output = os.path.join(ROOT_DIR, metric_csv_files, file_name + '_0.csv')
        feature_metrics.metrics.to_csv(metric_feature_output)

        hand_output_path = os.path.join(output_folder, thermal_image_name)
        cv2.imwrite(hand_output_path, hr.end_process)
        rm.add_metric(thermal_image_name, "number_points", len(all_points))
        rm.add_metric(thermal_image_name, "out_path_picture", pathlib.Path(hand_output_path).as_uri())
        rm.add_metric(thermal_image_name, "metric_feature_output", pathlib.Path(metric_feature_output).as_uri())


        feature_metrics.init_metric()
        os.remove(os.path.join(extracted_images, thermal_image_name))

        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print("Done")


def calculate_threshold_val(thermal_image):
    import numpy
    mean = numpy.ndarray.mean(thermal_image)
    if mean < 30:
        return 150
    if mean > 31.56:
        return 180
    if mean > 32.2:
        return 220
    return 150


def gather_all_points(hr):
    all_points = {}
    all_points.update(hr.feature_coordinates)
    all_points.update(hr.extra_points)
    all_points['center'] = hr.center
    return all_points


def get_thermal_file_type(color_map):
    if color_map == 0:
        return '.'
    elif color_map == 1:
        return '_thermal_bwr.'
    elif color_map == 2:
        return '_thermal_gist_ncar.'
    elif color_map == 3:
        return '_thermal_gnuplot2.'
    elif color_map == 4:
        return '_thermal_get.'

    return '.'














import os

import flirimageextractor
from matplotlib import cm

class FlirExtractorUtil():

    def __init__(self, thermal_images_path):
        self.thermal_images_path = thermal_images_path
        self.flir = flirimageextractor.FlirImageExtractor(exiftool_path='C:\\Program Files (x86)\\ExifTool\\ExifTool.exe',
                                                          # palettes=[cm.gnuplot2,cm.jet, cm.bwr, cm.gist_ncar], is_debug=True)
                                                          palettes=[cm.gnuplot2], is_debug=True)
                                             # palettes=[cm.jet, cm.bwr, cm.gist_ncar], is_debug=True)

    def get_thermal_image(self, file_name, extract_embedded_image=True):
        self.flir.process_image(os.path.join(self.thermal_images_path, file_name), extract_embedded_image)
        thermal_img = self.flir.extract_thermal_image()

        return thermal_img

    def extract_thermal_img_from_raw_img(self, file_name):
        # self.flir.loadfile(os.path.join(self.thermal_images_path, file_name))
        self.flir.process_image(os.path.join(self.thermal_images_path, file_name),True)
        self.flir.save_images()
        self.flir.plot()

import cv2
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
import libs.method.QcImage as QcImage
import libs.method.SICCalibrationRegression_MB3 as SICCalibrationRegression_MB3
from libs.model.TrainingSet import TrainingSet


JSON_PATH = 'Dataset/data_color_chart/tags.json'
IMAGE_PATH = 'Dataset/data_color_chart/'

RECT_SCALE = 1000

if __name__ == "__main__":

    jsonPath = JSON_PATH
    imagePath = IMAGE_PATH
    vis = False
    channel = 'green'

    # train
    with open(jsonPath) as json_data:
        objs = json.load(json_data)

    images_b = None
    images_g = None
    images_r = None

    for obj in objs:
        colors_b = []
        colors_g = []
        colors_r = []

        trainingSet = TrainingSet(obj)

        cv_image = cv2.imread(
            imagePath + trainingSet.imagePath, cv2.IMREAD_COLOR)

        if cv_image is None:
            print('Training image: ' + trainingSet.imagePath + ' cannot be found.')
            continue

        dis_image = cv_image.copy()

        height, width, channels = cv_image.shape

        background_anno = trainingSet.background

        background_area = QcImage.crop_image_by_position_and_rect(
            cv_image, background_anno.position, background_anno.rect)
        background_bgr = QcImage.get_average_rgb(background_area)

        colors_b.append(background_bgr[0])
        colors_g.append(background_bgr[1])
        colors_r.append(background_bgr[2])

        for anno in trainingSet.references:
            colour_area = QcImage.crop_image_by_position_and_rect(
                cv_image, anno.position, anno.rect)
            sample_bgr = QcImage.get_average_rgb(colour_area)

            colors_b.append(sample_bgr[0])
            colors_g.append(sample_bgr[1])
            colors_r.append(sample_bgr[2])

            # draw training label
            if vis:
                pos_x = int(width * anno.position.x)
                pos_y = int(height * anno.position.y)
                dim_x = int(width * anno.rect.x / RECT_SCALE) + pos_x
                dim_y = int(height * anno.rect.y / RECT_SCALE) + pos_y
                cv2.rectangle(dis_image,
                              (pos_x, pos_y),
                              (dim_x, dim_y),
                              (0, 255, 0), 1)

        images_b = np.array([colors_b]) if images_b is None else np.append(
            images_b, [colors_b], axis=0)
        images_g = np.array([colors_g]) if images_g is None else np.append(
            images_g, [colors_g], axis=0)
        images_r = np.array([colors_r]) if images_r is None else np.append(
            images_r, [colors_r], axis=0)

        # display training image and label
        if vis:
            dis_image = cv2.cvtColor(dis_image, cv2.COLOR_BGR2RGB)

            plt.imshow(dis_image)
            plt.title(trainingSet.imagePath)
            plt.show()

    if 'blue' in channel:
        # blue channel
        print('blue============')

        M_b, B_b, err_b = SICCalibrationRegression_MB3.sic_calibration_regression(
            images_b)
        print('a, b and error for blue channel: %s,%s, %s' %
              (M_b, B_b, err_b))

    if 'green' in channel:
        # green channel
        print('green============')

        M_g, B_g, err_g = SICCalibrationRegression_MB3.sic_calibration_regression(
            images_g)
        print('a, b and error for green channel: %s,%s, %s' %
              (M_g, B_g, err_g))

    if 'red' in channel:
        # red channel
        print('red============')

        M_r, B_r, err_r = SICCalibrationRegression_MB3.sic_calibration_regression(
            images_r)
        print('a, b and error for red channel: %s,%s, %s' %
              (M_r, B_r, err_r))

    input("Press Enter to exit...")

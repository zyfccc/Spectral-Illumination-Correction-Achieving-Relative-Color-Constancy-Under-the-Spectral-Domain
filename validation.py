import cv2
import json
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import libs.method.QcImage as QcImage
import libs.method.MathUtil as util
import libs.method.SICCalibrationRegression_MB3 as SICCalibrationRegression_MB3
from libs.model.TrainingSet import TrainingSet
from libs.model.Num3 import Num3

# Test and visualize nonuniform illumination correction and illumination matching algorithms on a single image

JSON_PATH = 'Dataset/data_nonuniform_illumination/tags.json'
IMAGE_PATH = 'Dataset/data_nonuniform_illumination/'

RECT_SCALE = 1000

bg_tag = 0


def nonuniform_illumination_correction(img, bg_img):
    img = img / 255.0
    bg_img = bg_img / 255.0
    image = QcImage.spectral_nonuniform_illumination_correction_pixel(
        bg_img, img)
    image *= 255.0
    image = image.astype(np.uint8)
    return image


def colour_matching(image, image_bg_bgr, bg_bgr):
    return QcImage.fast_spectral_illumination_matching_pixel(image, image_bg_bgr, bg_bgr)


def generate_background_image(image):
    background_image = QcImage.morphology_close_background(
        image, 5, 91)
    return QcImage.gaussian_blur_background(
        background_image, 5, 41)


if __name__ == "__main__":

    jsonPath = JSON_PATH
    imagePath = IMAGE_PATH
    vis = True
    only_one = False

    # train
    with open(jsonPath) as json_data:
        objs = json.load(json_data)

    obj = objs[0]

    colors_b_center_ori = []
    colors_g_center_ori = []
    colors_r_center_ori = []

    colors_b_corner_ori = []
    colors_g_corner_ori = []
    colors_r_corner_ori = []

    colors_b_center_sic = []
    colors_g_center_sic = []
    colors_r_center_sic = []

    colors_b_corner_sic = []
    colors_g_corner_sic = []
    colors_r_corner_sic = []

    colors_b_center_sic2 = []
    colors_g_center_sic2 = []
    colors_r_center_sic2 = []

    angle1 = []
    angle2 = []
    angle3 = []

    dist1 = []
    dist2 = []
    dist3 = []

    trainingSet = TrainingSet(obj)

    cv_image = cv2.imread(
        imagePath + trainingSet.imagePath, cv2.IMREAD_COLOR)

    for i in range(9):
        anno = trainingSet.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            cv_image, anno.position, anno.rect)
        sample_bgr1 = QcImage.get_average_rgb(colour_area)

        colors_b_center_ori.append(sample_bgr1[0])
        colors_g_center_ori.append(sample_bgr1[1])
        colors_r_center_ori.append(sample_bgr1[2])

        if not only_one:
            anno = trainingSet.references[i + 9]
            colour_area = QcImage.crop_image_by_position_and_rect(
                cv_image, anno.position, anno.rect)
            sample_bgr2 = QcImage.get_average_rgb(colour_area)

            colors_b_corner_ori.append(sample_bgr2[0])
            colors_g_corner_ori.append(sample_bgr2[1])
            colors_r_corner_ori.append(sample_bgr2[2])

            angle1.append(util.angle(sample_bgr1, sample_bgr2))
            dist1.append(util.rmse(sample_bgr1, sample_bgr2))

    '''print("original r center are: " +
          str(["%.1f" % elem for elem in colors_r_center_ori]))
    print("original r corner are: " +
          str(["%.1f" % elem for elem in colors_r_corner_ori]))

    print("original g center are: " +
          str(["%.1f" % elem for elem in colors_g_center_ori]))
    print("original g corner are: " +
          str(["%.1f" % elem for elem in colors_g_corner_ori]))

    print("original b center are: " +
          str(["%.1f" % elem for elem in colors_b_center_ori]))
    print("original b corner are: " +
          str(["%.1f" % elem for elem in colors_b_corner_ori]))'''

    # Color performance without any correction

    print("rms 1 error is: " + str(dist1))
    print("rms 1 error is: " + str(sum(dist1)))

    print("Angle error 1 is: " + str(angle1))
    print("Angle error 1 is: " + str(sum(angle1)))

    bg_image = generate_background_image(cv_image)

    bg_bgr_image1 = None

    corrected_image = cv_image

    corrected_image = nonuniform_illumination_correction(cv_image, bg_image)

    height, width, channels = cv_image.shape

    for i in range(9):
        anno = trainingSet.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            corrected_image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)

        colors_b_center_sic.append(sample_bgr[0])
        colors_g_center_sic.append(sample_bgr[1])
        colors_r_center_sic.append(sample_bgr[2])

        if i == bg_tag:
            bg_rgb_image1 = sample_bgr

        if not only_one:
            anno = trainingSet.references[i + 9]
            colour_area = QcImage.crop_image_by_position_and_rect(
                corrected_image, anno.position, anno.rect)
            sample_bgr = QcImage.get_average_rgb(colour_area)

            colors_b_corner_sic.append(sample_bgr[0])
            colors_g_corner_sic.append(sample_bgr[1])
            colors_r_corner_sic.append(sample_bgr[2])

            if i + 9 == bg_tag:
                bg_rgb_image1 = sample_bgr

            dist2.append(util.rmse(np.array([colors_b_center_sic[i], colors_g_center_sic[i], colors_r_center_sic[i]]),
                                   np.array([colors_b_corner_sic[i], colors_g_corner_sic[i], colors_r_corner_sic[i]])))
            angle2.append(util.angle([colors_b_center_sic[i], colors_g_center_sic[i], colors_r_center_sic[i]],
                                     [colors_b_corner_sic[i], colors_g_corner_sic[i], colors_r_corner_sic[i]]))

    # Performance of nonuniform illumination correction

    print("rms 2 error is: " + str(dist2))
    print("rms 2 error is: " + str(sum(dist2)))

    print("Angle error 2 is: " + str(angle2))
    print("Angle error 2 is: " + str(sum(angle2)))

    dis_image = corrected_image.copy()

    if vis:
        dis_image = cv2.cvtColor(dis_image, cv2.COLOR_BGR2RGB)

        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

        plt.imshow(dis_image)
        plt.title(trainingSet.imagePath)
        plt.axis('off')
        plt.show()
        plt.savefig('snic.png', dpi=300)

    '''print("sic r center are: " +
          str(["%.1f" % elem for elem in colors_r_center_sic]))
    print("sic r corner are: " +
          str(["%.1f" % elem for elem in colors_r_corner_sic]))

    print("sic g center are: " +
          str(["%.1f" % elem for elem in colors_g_center_sic]))
    print("sic g corner are: " +
          str(["%.1f" % elem for elem in colors_g_corner_sic]))

    print("sic b center are: " +
          str(["%.1f" % elem for elem in colors_b_center_sic]))
    print("sic b corner are: " +
          str(["%.1f" % elem for elem in colors_b_corner_sic]))'''

    if not only_one:
        rmse_r = util.rmse(np.array(colors_r_center_sic),
                           np.array(colors_r_corner_sic))
        rmse_g = util.rmse(np.array(colors_g_center_sic),
                           np.array(colors_g_corner_sic))
        rmse_b = util.rmse(np.array(colors_b_center_sic),
                           np.array(colors_b_corner_sic))

        mean = np.mean(np.array([rmse_r, rmse_g, rmse_b]))
        # print("sic rms error is: " + str(mean))

    ###################

    anno = trainingSet.references[bg_tag]
    colour_area = QcImage.crop_image_by_position_and_rect(
        corrected_image, anno.position, anno.rect)
    sample_bgr = QcImage.get_average_rgb(colour_area)

    corrected_image = colour_matching(
        corrected_image, sample_bgr, bg_rgb_image1)

    if vis:
        dis_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)

        plt.imshow(dis_image)
        plt.title(trainingSet.imagePath)
        plt.axis('off')
        plt.show()

    for i in range(9):
        anno = trainingSet.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            corrected_image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)

        colors_b_center_sic2.append(sample_bgr[0])
        colors_g_center_sic2.append(sample_bgr[1])
        colors_r_center_sic2.append(sample_bgr[2])

        dist3.append(util.rmse(sample_bgr, np.array([
            colors_b_center_sic[i], colors_g_center_sic[i], colors_r_center_sic[i]])))
        angle3.append(util.angle(sample_bgr, np.array([
                      colors_b_center_sic[i], colors_g_center_sic[i], colors_r_center_sic[i]])))

    # Performance of nonuniform illumination correction + illumination matching

    print("rms 3 error is: " + str(dist3))
    print("rms 3 error is: " + str(sum(dist3)))

    print("Angle error 3 is: " + str(angle3))
    print("Angle error 3 is: " + str(sum(angle3)))


input("Press Enter to exit...")

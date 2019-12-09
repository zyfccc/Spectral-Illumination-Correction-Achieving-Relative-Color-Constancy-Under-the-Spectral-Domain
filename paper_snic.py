import cv2
import json
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import libs.method.QcImage as QcImage
import libs.method.MathUtil as util
from libs.model.TrainingSet import TrainingSet
from libs.model.Num3 import Num3

# Test and visualize nonuniform illumination correction algorithms on a image dataset

JSON_PATH = 'Dataset/data_nonuniform_illumination/tags.json'
IMAGE_PATH = 'Dataset/data_nonuniform_illumination/'

RECT_SCALE = 1000
A_BGR = [15.2, 15.2, 15.2]
B_BGR = [1.0, 1.0, 1.0]
GAMMA_BGR = [2.4, 2.4, 2.4]
LOG_BASE = [10.0, 10.0, 10.0]

bg_tag = 0


def nonuniform_illumination_correction(img, bg_img):
    # return QcImage.retinex_pde(img)
    return QcImage.spectral_nonuniform_illumination_correction_pixel(bg_img, img, a_bgr=A_BGR, b_bgr=B_BGR, gamma_bgr=GAMMA_BGR, logbase=LOG_BASE)
    # return QcImage.retinex_with_adjust(img)
    # return QcImage.illumination_correction_lab(bg_img, img)
    # return img


def generate_background_image(image):
    background_image = QcImage.morphology_close_background(
        image, 5, 91)
    return QcImage.gaussian_blur_background(
        background_image, 5, 41)


if __name__ == "__main__":

    jsonPath = JSON_PATH
    imagePath = IMAGE_PATH
    vis = False
    count = 0
    dists = None

    # train
    with open(jsonPath) as json_data:
        objs = json.load(json_data)

    for obj in objs:

        colors_b_center_sic = []
        colors_g_center_sic = []
        colors_r_center_sic = []

        colors_b_corner_sic = []
        colors_g_corner_sic = []
        colors_r_corner_sic = []

        dists_temp = []

        trainingSet = TrainingSet(obj)

        cv_image = cv2.imread(
            imagePath + trainingSet.imagePath, cv2.IMREAD_COLOR)

        # bg_image = cv2.imread(
        #    imagePath + "background1.jpg", cv2.IMREAD_COLOR)
        bg_image = generate_background_image(cv_image)

        bg_bgr_image1 = None

        corrected_image = cv_image

        corrected_image = nonuniform_illumination_correction(
            cv_image, bg_image)

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

            anno = trainingSet.references[i + 9]
            colour_area = QcImage.crop_image_by_position_and_rect(
                corrected_image, anno.position, anno.rect)
            sample_bgr = QcImage.get_average_rgb(colour_area)

            colors_b_corner_sic.append(sample_bgr[0])
            colors_g_corner_sic.append(sample_bgr[1])
            colors_r_corner_sic.append(sample_bgr[2])

            if i + 9 == bg_tag:
                bg_rgb_image1 = sample_bgr

            dists_temp.append(util.rmse(np.array([colors_b_center_sic[i], colors_g_center_sic[i], colors_r_center_sic[i]]),
                                        np.array([colors_b_corner_sic[i], colors_g_corner_sic[i], colors_r_corner_sic[i]])))

        print("rms error of " + str(count) + " is: " + str(sum(dists_temp)))
        if dists is None:
            dists = np.array(np.array([dists_temp]))
        else:
            dists = np.concatenate((dists, np.array([dists_temp])), axis=0)

        #corrected_image = QcImage.grey_world(corrected_image)

        dis_image = corrected_image.copy()

        # display training image and label
        if vis:
            dis_image = cv2.cvtColor(dis_image, cv2.COLOR_BGR2RGB)

            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300

            plt.imshow(dis_image)
            plt.title(trainingSet.imagePath)
            plt.axis('off')
            plt.show()
            #plt.savefig('sic.png', dpi=300)

        count = count + 1

    print("Total rmse of " + str(count) + " objects is: " + str(np.sum(dists)))
    print(str(np.mean(dists, axis=0).tolist()))


input("Press Enter to exit...")

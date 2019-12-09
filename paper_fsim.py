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


JSON_PATH = 'Dataset/data_color_chart/tags.json'
IMAGE_PATH = 'Dataset/data_color_chart/'

RECT_SCALE = 1000
A_BGR = [15.2, 15.2, 15.2]
B_BGR = [1.0, 1.0, 1.0]
GAMMA_BGR = [2.4, 2.4, 2.4]
LOG_BASE = [10.0, 10.0, 10.0]

vis = False


def colour_matching(image, image_bg_bgr, bg_bgr):
    # return image
    # return QcImage.grey_world(image)
    return QcImage.fast_spectral_illumination_matching_pixel(image, image_bg_bgr, bg_bgr, a_bgr=A_BGR, b_bgr=B_BGR, gamma_bgr=GAMMA_BGR, logbase=LOG_BASE)


# match obj2 to obj1
def illumination_match(obj1, obj2):

    dists_temp = []

    trainingSet1 = TrainingSet(obj1)
    cv_image1 = cv2.imread(
        IMAGE_PATH + trainingSet1.imagePath, cv2.IMREAD_COLOR)

    trainingSet2 = TrainingSet(obj2)
    cv_image2 = cv2.imread(
        IMAGE_PATH + trainingSet2.imagePath, cv2.IMREAD_COLOR)

    background_anno = trainingSet1.references[15]
    background_area = QcImage.crop_image_by_position_and_rect(
        cv_image1, background_anno.position, background_anno.rect)
    background_bgr1 = QcImage.get_average_rgb(background_area)

    background_anno = trainingSet2.references[15]
    background_area = QcImage.crop_image_by_position_and_rect(
        cv_image2, background_anno.position, background_anno.rect)
    background_bgr2 = QcImage.get_average_rgb(background_area)

    background_bgr3 = colour_matching(
        np.array([[background_bgr2]]), background_bgr2, background_bgr1)

    # background_area = QcImage.crop_image_by_position_and_rect(
    #     matched_image, background_anno.position, background_anno.rect)
    # background_bgr2 = QcImage.get_average_rgb(background_area)

    dists_temp.append(util.rmse(background_bgr1, background_bgr3))

    for i in range(len(trainingSet2.references)):
        anno = trainingSet1.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            cv_image1, anno.position, anno.rect)
        sample_bgr1 = QcImage.get_average_rgb(colour_area)

        anno = trainingSet2.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            cv_image2, anno.position, anno.rect)
        sample_bgr2 = QcImage.get_average_rgb(colour_area)

        sample_bgr2 = colour_matching(
            np.array([[sample_bgr2]]), background_bgr2, background_bgr1)

        error = util.rmse(sample_bgr1, sample_bgr2)

        dists_temp.append(error)

    # if vis:
    #     dis_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

    #     # plt.rcParams['figure.dpi'] = 300
    #     # plt.rcParams['savefig.dpi'] = 300

    #     plt.imshow(dis_image)
    #     plt.title(trainingSet2.imagePath)
    #     plt.axis('off')
    #     plt.show()
    #     # plt.savefig('sic.png', dpi=300)

    return dists_temp


if __name__ == "__main__":

    count = 0
    dists = None

    # train
    with open(JSON_PATH) as json_data:
        objs = json.load(json_data)

    for obj1 in objs:
        for obj2 in objs:
            if obj1 == obj2:
                continue

            errors = illumination_match(obj1, obj2)

            print("rms error of " + str(count) + " is: " + str(sum(errors)))
            if dists is None:
                dists = np.array([errors])
            else:
                dists = np.concatenate((dists, np.array([errors])), axis=0)

            count = count + 1

    print("Total rmse of " + str(count) + " objects is: " + str(np.sum(dists)))
    print(str(np.mean(dists, axis=0).tolist()))


input("Press Enter to exit...")

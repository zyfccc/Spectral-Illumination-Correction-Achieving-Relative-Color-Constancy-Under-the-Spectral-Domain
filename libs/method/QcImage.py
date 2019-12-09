import io
import cv2
import numpy as np
import math
from scipy.optimize import least_squares


RECT_SCALE = 1000

bg_R = np.array([150.0, 150.0, 150.0])


def get_average_rgb(image_data):
    return np.average(image_data, axis=(0, 1))


def crop_image_by_position_and_rect(cv_image, position, rect):
    # img[y: y + h, x: x + w]
    height = cv_image.shape[0]
    width = cv_image.shape[1]
    position_x = position.x * width
    position_y = position.y * height
    rect_x = width * rect.x / RECT_SCALE
    rect_y = height * rect.y / RECT_SCALE
    return cv_image[int(position_y):int(position_y) + int(rect_y),
                    int(position_x):int(position_x) + int(rect_x)]


def spectral_nonuniform_illumination_correction_pixel(background, target, a_bgr, b_bgr, gamma_bgr, logbase):
    x, y, c = background.shape
    colors_background = np.reshape(background, (x * y, 3))
    colors_target = np.reshape(target, (x * y, 3))
    gamma_corrected_target = np.power(colors_target, np.divide(1, gamma_bgr))
    gamma_corrected_background = np.power(
        colors_background, np.divide(1, gamma_bgr))
    multiplier = np.multiply(bg_R, b_bgr)
    exponent = np.divide(np.subtract(
        gamma_corrected_target, gamma_corrected_background), a_bgr)
    power = np.power(logbase, exponent)
    final = np.multiply(multiplier, power)
    clipped = np.clip(final, 0.0, 255.0)
    image = np.reshape(clipped, (x, y, 3))
    image = image.astype(np.uint8)
    return image


def fast_spectral_illumination_matching_pixel(cv_image, cv_image_bg, background_rgb, a_bgr, b_bgr, gamma_bgr, logbase):

    height, width, channels = cv_image.shape
    result_image = cv_image.copy()

    for y in range(0, height):
        for x in range(0, width):
            for c in range(0, channels):
                bg_rgb_clone = cv_image_bg.copy()
                for N in range(3):
                    target_pixel = result_image.item(y, x, c)
                    result_image.itemset((y, x, c), fast_luminance_normalization_pixel(
                        target_pixel, background_rgb[c], bg_rgb_clone[c], a_bgr[c], b_bgr[c], gamma_bgr[c], logbase[c]))
                    bg_rgb_clone[c] = fast_luminance_normalization_pixel(
                        bg_rgb_clone[c], background_rgb[c], bg_rgb_clone[c], a_bgr[c], b_bgr[c], gamma_bgr[c], logbase[c])
                    if abs(bg_rgb_clone[c] - background_rgb[c]) < 1.0 or bg_rgb_clone[c] >= 255.0:
                        break

    return result_image


def fast_luminance_normalization_pixel(dt2, db1, db2, M, B, D, LOG):
    dt1 = 0.0
    dt2_gamma = np.power(dt2, 1 / D)
    db1_gamma = np.power(db1, 1 / D)
    db2_gamma = np.power(db2, 1 / D)
    try:
        delta_lum_back = (math.pow(LOG, db1_gamma / M) -
                          math.pow(LOG, db2_gamma / M)) / db1
        pixel = M * math.log(dt2 * B * delta_lum_back + math.pow(
            LOG, dt2_gamma / M), LOG)
        pixel = np.power(pixel, D)

        # clamp pixel within [0, 255]
        dt1 = max(0.0, min(pixel, 255.0))
    except:
        dt1 = 255
    return dt1


def image_preprocessing(cv_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
    cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
    return cv_image


def gaussian_blur_background(cv_image, repeat=1, size=31):
    for x in range(5):
        cv_image = cv2.GaussianBlur(cv_image, (size, size), 0)
    return cv_image


def morphology_close_background(cv_image, repeat=1, size=31):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    for x in range(repeat):
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
    return cv_image

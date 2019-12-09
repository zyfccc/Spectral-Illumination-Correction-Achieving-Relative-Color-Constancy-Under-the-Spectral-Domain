import sys
import math
import cv2
import logging
import tensorflow as tf
import numpy as np
import libs.method.QcImage as QcImage
import matplotlib.pyplot as plt
from libs.model.Num2 import Num2

random_init_num = 200

means = None

WEIGHT = 1000


def sic_calibration_regression(values, learning_rate=1E-2, escape_rate=1E-6):

    # random initialization
    a = tf.Variable(tf.random_uniform([1], 0.2, 500), name='a')
    #b = tf.Variable(tf.random_normal([1], 2.0, 2.0), name='b')
    c = tf.Variable(tf.random_uniform([1], 2.0, 2.0), name='c')

    x, y = values.shape

    means = np.mean(values, axis=0)
    target_bg_value = np.empty([x, y])
    background_value = np.empty([x, y])
    target_bg_value[:, :] = means[0]
    for i in range(x):
        background_value[i, :] = values[i, 0]

    input_value = tf.placeholder(tf.float32, shape=[x, y])
    input_background = tf.placeholder(tf.float32, shape=[x, y])
    input_target_bg = tf.placeholder(tf.float32, shape=[x, y])

    # cost function
    cost_function1 = loss_function1(
        input_value, input_background, input_target_bg, a, 2.4)

    # training algorithm
    optimizer1 = tf.train.AdamOptimizer(
        learning_rate).minimize(cost_function1)

    res_a = None
    res_b = None
    min_error = sys.maxsize

    for j in range(random_init_num):

        # initializing the variables
        init = tf.global_variables_initializer()

        # starting the session session
        sess = tf.Session()
        sess.run(init)

        epoch = 10000
        prev_training_cost = sys.maxsize

        for step in range(epoch):
            _, training_cost = sess.run(
                [optimizer1, cost_function1], feed_dict={
                    input_value: values,
                    input_background: background_value,
                    input_target_bg: target_bg_value})

            if math.isinf(training_cost) or math.isnan(training_cost):
                break

            if np.abs(prev_training_cost - training_cost) < escape_rate:
                break

            prev_training_cost = training_cost

        if min_error > prev_training_cost:
            res_a = sess.run(a)[0].item()
            res_b = sess.run(c)[0].item()
            min_error = prev_training_cost

            print a.eval(session=sess)
            # print c.eval(session=sess)
            print min_error
            print '========='

    return res_a, res_b, min_error


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def fsim(a, b, B1, B2, T2):
    first_exponent = tf.div(tf.pow(B1, 1 / b), a)
    first_item = tf.pow(10.0, first_exponent)
    second_exponent = tf.div(tf.pow(B2, 1 / b), a)
    second_item = tf.pow(10.0, second_exponent)
    first_half = tf.multiply(T2, tf.div(
        tf.subtract(first_item, second_item), B1))

    fourth_exponent = tf.div(tf.pow(T2, 1 / b), a)
    fourth_item = tf.pow(10.0, fourth_exponent)

    return tf.pow(tf.multiply(a, log10(tf.add(first_half, fourth_item))), b)


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def loss_function1(values, backgrounds, target_bgs, a, b):
    #target_bgs = tf.Print(target_bgs, [target_bgs], "target_bgs")
    i = tf.constant(0)
    #a = tf.Print(a, [a], "a")
    #b = tf.Print(b, [b], "b")

    def condition1(i, values, backgrounds): return tf.less(i, 3)

    def body1(i, values, backgrounds):
        # update pixel value
        #backgrounds = tf.Print(backgrounds, [backgrounds], "backgrounds")
        # with tf.control_dependencies([backgrounds]):
        #values = tf.Print(values, [values], "values")
        values = fsim(a, b, target_bgs, backgrounds, values)
        #op1 = tf.Print(op1, [op1], "op1")
        op1 = tf.clip_by_value(values, 0, 255)
        with tf.control_dependencies([op1]):
            backgrounds = fsim(a, b, target_bgs, backgrounds, backgrounds)
            # update background
            op2 = tf.clip_by_value(backgrounds, 0, 255)
            with tf.control_dependencies([op2]):
                return tf.identity(i + 1), op1, op2

    i, values, backgrounds = tf.while_loop(
        condition1, body1, loop_vars=[i, values, backgrounds])

    with tf.control_dependencies([i]):
        #values = tf.Print(values, [values], 'values', summarize=50)
        _, light_variance = tf.nn.moments(values, [0])
        _, color_variance = tf.nn.moments(values, [1])

        # light_variance = tf.Print(
        #    light_variance, [light_variance], summarize=50)
        #_ = tf.Print(
        #    _, [_], 'mean', summarize=50,)

        result = tf.global_norm(
            [light_variance, WEIGHT / color_variance])

        # light_variance = tf.Print(
        #    light_variance, [light_variance], "light variance")
        # color_variance = tf.Print(
        #    color_variance, [color_variance], "color variance")

        return result

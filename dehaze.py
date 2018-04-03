#!/bin/env python
#
# This is the Dark Channel Prior method for dehazing process.
#
# Author: Wenrui Zhang
# Date: Apr 1, 2018

import os
import sys
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim

# parameters are set here
PATCH_SIZE = 15
HAZE_WEIGHT = 0.95
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
IMG_START = 1
IMG_END = 2


def dark_channel(input_img, patch_size):
    b, g, r = cv2.split(input_img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dcp = cv2.erode(dc, kernel)
    return dcp


def atm_light(img, dcp):
    [h, w] = img.shape[:2]
    img_size = h*w
    num_pixel = int(max(math.floor(img_size*BRIGHTEST_PIXELS_PERCENTAGE), 1))
    dark_channel_vec = dcp.reshape(img_size)
    img_vec = img.reshape(img_size, 3)

    indices = dark_channel_vec.argsort()
    indices = indices[img_size-num_pixel::]

# highest intensity in the input image I are selected as the atmospheric light.
    brightest_pixel = img_vec[indices]
    brightest_r = brightest_pixel[:, 0]
    brightest_g = brightest_pixel[:, 1]
    brightest_b = brightest_pixel[:, 2]
    A = np.zeros(3)
    A[0] = max(brightest_r)
    A[1] = max(brightest_g)
    A[2] = max(brightest_b)

# average form of A
#    atm_sum = np.zeros(3)
#    for ind in range(1, num_pixel):
#       atm_sum = atm_sum + img_vec[indices[ind]]
#    A = atm_sum / num_pixel

    return A


def transmission_estimate(im, A, sz):
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind]/A[ind]

    transmission = 1 - HAZE_WEIGHT*dark_channel(im3, sz)
    return transmission


def guided_filter(im, p, r, eps):
    mean_i = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
    cov_ip = mean_ip - mean_i*mean_p

    mean_ii = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_i = mean_ii - mean_i*mean_i

    a = cov_ip/(var_i + eps)
    b = mean_p - a*mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a*im + mean_b
    return q


def transmission_refine(im, estimate_t):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, estimate_t, r, eps)
    return t


def recover(im, t_estimate, A, t_bound=0.1):
    res = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[ind])/cv2.max(t_estimate, t_bound) + A[ind]
    return res


def psnr(original_img, recovered_img):
    [h, w] = original_img.shape[:2]
    img_size = h * w
    original_img_vec = original_img.reshape(img_size, 3)
    recovered_img_vec = recovered_img.reshape(img_size, 3)
    psnr_all_channel = 0
    for i in range(0, 3):
        diff = original_img_vec[:, i] - recovered_img_vec[:, i]
        psnr_single_channel = 20 * math.log10(1/np.std(diff))
        psnr_all_channel = psnr_all_channel+psnr_single_channel
    return psnr_all_channel/3


if __name__ == '__main__':
    # if images file SOTS is not at current directory, please change the data_dir
    data_dir = './SOTS'
    in_or_out = 'outdoor'
    im_dir = os.path.join(data_dir, in_or_out, 'hazy')
    ref_dir = os.path.join(data_dir, in_or_out, 'gt')

    def nothing(*argv):
        pass


    if not os.path.isdir(im_dir):
        print('Given path {} not found'.format(im_dir))
        sys.exit(-1)

    img_read = 0
    sum_ssim = 0
    sum_psnr = 0
    img = 0
    ref_img = 0
    recovered_img = 0

    for fn in os.listdir(im_dir):
        if img_read+1 < IMG_START:
            continue
        if img_read+1 >= IMG_END:
            break
        img_read = img_read + 1
        im_path = os.path.join(im_dir, fn)
        fn_ref = fn[:4]+'.png'
        ref_path = os.path.join(ref_dir, fn_ref)
        assert(os.path.exists(im_path)),'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(ref_path)),'Annotation: {} does not exist'.format(ref_path)
        ref_src = cv2.imread(ref_path)
        ref_img = ref_src.astype('float64')/255
        src = cv2.imread(im_path)
        img = src.astype('float64')/255

        dark = dark_channel(img, PATCH_SIZE)
        A = atm_light(img, dark)
        t_estimated = transmission_estimate(img, A, PATCH_SIZE)
        t_refined = transmission_refine(src, t_estimated)
        recovered_img = recover(img, t_refined, A, 0.1)
        ssim_val = ssim(recovered_img, ref_img, multichannel=True)
        psnr_val = psnr(recovered_img, ref_img)
        sum_ssim = sum_ssim+ssim_val
        sum_psnr = sum_psnr+psnr_val

    avg_psnr = sum_psnr/img_read
    avg_ssim = sum_ssim/img_read

    print('%d images are tested' % img_read)
    print('The average PSNR value is %0.4f' % avg_psnr)
    print('The average SSIM value is %0.4f' % avg_ssim)

    cv2.imshow('Original Image', img)
    cv2.imshow('Recovered Image', recovered_img)
    cv2.imshow('Reference Image', ref_img)
    cv2.waitKey()

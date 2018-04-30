import os
import sys
import cv2
import math
import numpy as np
import Sky_Detection
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

# Parameters defined here
LARGE_PATCH_SIZE = 15
SMALL_PATCH_SIZE = 3
HAZE_WEIGHT = 0.95
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
PIXEL_MAX = 255.0

# Function define here
DETECT_SKY = 1
PRIOR_OPTIMIZE = 0
RECOVER_REFINE = 1
RECOVER_WHITENING = 1
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 1
TRAIN_METHOD = 1
ENABLE_PRINT_FINAL_IMAGE = 0

METHOD = 1  # 1: DCP; 2: test; 3: DCP Weight test; 4: Image information print out

FOLDS = 5
ITER_NUM = 5


# DCP starts here
def dark_channel_prior(input_img, sky):
    dc = dark_channel(input_img)
    [h, w] = input_img.shape[:2]
    if PRIOR_OPTIMIZE == 1 and np.sum(sky.reshape(h*w)) > h*w*0.05:
        assert(DETECT_SKY > 0)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (LARGE_PATCH_SIZE, LARGE_PATCH_SIZE))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (SMALL_PATCH_SIZE, SMALL_PATCH_SIZE))
        dcp_large = cv2.erode(dc, kernel_large)
        dcp_small = cv2.erode(dc, kernel_small)
        dcp = np.zeros(input_img.shape[:2])
        for dcp_i in range(h):
            for dcp_j in range(w):
                x = dcp_i-2
                y = dcp_j-2
                if dcp_i < 2:
                    x = 0
                if dcp_j < 2:
                    y = 0
                if dcp_i >= h-6:
                    x = h-6
                if dcp_j >= w-6:
                    y = w-6
                region = sky[x:x+5, y:y+5]
                [r_h, r_w] = region.shape
                sum_region = sum(region.reshape(r_h*r_w))
                if 8 < sum_region < 18:
                    dcp[dcp_i, dcp_j] = dcp_small[dcp_i, dcp_j]
                else:
                    dcp[dcp_i, dcp_j] = dcp_large[dcp_i, dcp_j]
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LARGE_PATCH_SIZE, LARGE_PATCH_SIZE))
        dcp = cv2.erode(dc, kernel)
    return dcp


def dark_channel(input_img):
    b, g, r = cv2.split(input_img)
    dc = cv2.min(cv2.min(r, g), b)
    return dc


def bright_channel(input_img):
    b, g, r = cv2.split(input_img)
    bc = cv2.max(cv2.max(r, g), b)
    return bc


def atm_light(im, sky):
    img_sky = im.copy()
    img_sky[sky == 0] = [0, 0, 0]
    img_non_sky = im.copy()
    [h, w] = im.shape[:2]
    img_size = h * w
    if sum(sky.reshape(img_size)) > img_size*0.05:
        if BRIGHTEST_IN_DC:
            dc = dark_channel_prior(img_sky, sky)
            num_pixel = int(max(math.floor(img_size * BRIGHTEST_PIXELS_PERCENTAGE), 1))
            dark_channel_vec = dc.reshape(img_size)
            img_vec = img_sky.reshape(img_size, 3)
            indices = (-dark_channel_vec).argsort()[:num_pixel]
            brightest_pixel = img_vec[indices]
            b = brightest_pixel[:, 0]
            g = brightest_pixel[:, 1]
            r = brightest_pixel[:, 2]
            brightest = np.argmax(np.power(b, 2) + np.power(g, 2) + np.power(r, 2))
            a = brightest_pixel[brightest]
        else:
            num_pixel = int(max(math.floor(img_size * BRIGHTEST_PIXELS_PERCENTAGE), 1))
            b = img_sky[:, :, 0].reshape(img_size)
            g = img_sky[:, :, 1].reshape(img_size)
            r = img_sky[:, :, 2].reshape(img_size)

            indices_b = (-b).argsort()[:num_pixel]
            indices_g = (-g).argsort()[:num_pixel]
            indices_r = (-r).argsort()[:num_pixel]
            # average form of a
            atm_sum = np.zeros(3)
            for ind in range(num_pixel):
                atm_sum[0] += b[indices_b[ind]]
                atm_sum[1] += g[indices_g[ind]]
                atm_sum[2] += r[indices_r[ind]]
            a = atm_sum / num_pixel
# for non sky
    else:
        if BRIGHTEST_IN_DC:
            dc = dark_channel_prior(img_non_sky, sky)
            num_pixel = int(max(math.floor(img_size * BRIGHTEST_PIXELS_PERCENTAGE), 1))
            dark_channel_vec = dc.reshape(img_size)
            img_vec = img_non_sky.reshape(img_size, 3)
            indices = (-dark_channel_vec).argsort()[:num_pixel]
            brightest_pixel = img_vec[indices]
            b = brightest_pixel[:, 0]
            g = brightest_pixel[:, 1]
            r = brightest_pixel[:, 2]
            brightest = np.argmax(np.power(b, 2) + np.power(g, 2) + np.power(r, 2))
            a = brightest_pixel[brightest]
        else:
            x0 = 0
            y0 = 0
            x1 = h - 1
            y1 = w - 1
            while (x1 - x0) * (y1 - y0) > 0.05 * img_size:
                x1 = int((x1 + x0) / 2)
                y1 = int((y1 + y0) / 2)
                x0, x1, y0, y1 = sub_block(img, x0, x1, y0, y1)
            block = img_non_sky[x0:x1, y0:y1]
            sky_new = sky[x0:x1, y0:y1]
            [h_b, w_b] = block.shape[:2]
            img_size = h_b * w_b
            dc = dark_channel_prior(block, sky_new)
            num_pixel = int(max(math.floor(img_size * 0.1), 1))
            dark_channel_vec = dc.reshape(img_size)
            img_vec = block.reshape(img_size, 3)
            indices = (-dark_channel_vec).argsort()[:num_pixel]
            brightest_pixel = img_vec[indices]
            b = brightest_pixel[:, 0]
            g = brightest_pixel[:, 1]
            r = brightest_pixel[:, 2]
            brightest = np.argmax(np.power(b, 2) + np.power(g, 2) + np.power(r, 2))
            a = brightest_pixel[brightest]
    return a


def sub_block(im, x0, x1, y0, y1):
    [h, w] = im.shape[:2]
    block_all = [[x0, x1, y0, y1], [x1, 2*x1-x0 if (2*x1-x0) < h else h, y0, y1],
                 [x0, x1, y1, 2*y1-y0 if (2*y1-y0) < w else w],
                 [x1, 2*x1-x0 if (2*x1-x0) < h else h, y1, 2*y1-y0 if (2*y1-y0) < w else w]]
    block_value = []
    for block_i in range(4):
        block_value.append(sub_block_value(img[block_all[block_i][0]:block_all[block_i][1],
                                           block_all[block_i][2]:block_all[block_i][3]]))
    index = int(np.argmax(block_value))
    return block_all[index][0], block_all[index][1], block_all[index][2],  block_all[index][3]


def sub_block_value(block):
    [h, w] = block.shape[:2]
    block_size = h * w
    block_vec = block.reshape(block_size, 3)
    block_sum = 0
    for ind in range(3):
        block_sum += np.abs(np.mean(block_vec[:, 0])-np.std(block_vec[:, 0]))
    return block_sum / 3


def transmission_estimate(im, a, sky, weight):
    im3 = np.empty(im.shape, im.dtype)
    [h, w] = im.shape[:2]
    img_size = h * w
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind]/a[ind]
    if INVERSE_SKY == 0:
        if weight is None:
            transmission = 1 - HAZE_WEIGHT * dark_channel_prior(im3, sky)
        else:
            transmission_sky = 1 - weight[0] * dark_channel_prior(im3, sky)
            transmission_non_sky = 1 - weight[1] * dark_channel_prior(im3, sky)
            transmission = transmission_sky
            transmission[sky == 0] = transmission_non_sky[sky == 0]
    else:
        if sum(sky.reshape(img_size)) > 0:
            transmission_non_sky = 1 - HAZE_WEIGHT * dark_channel_prior(im3, sky)
            img_inverse = 255 - im
            sky_inverse = sky.copy()
            sky_inverse += 1
            sky_inverse[sky_inverse > 1] = 0
            a_inverse = atm_light(img_inverse, sky_inverse)
            im3_inverse = np.empty(im.shape, im.dtype)
            for ind in range(3):
                im3_inverse[:, :, ind] = img_inverse[:, :, ind] / a_inverse[ind]
            transmission_sky = 1 - HAZE_WEIGHT * dark_channel_prior(im3_inverse, sky_inverse)
            transmission = transmission_non_sky
            transmission[sky > 0] = transmission_sky[sky > 0]
        else:
            transmission = 1 - HAZE_WEIGHT * dark_channel_prior(im3, sky)

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
    int_img = np.array(im, dtype=np.uint8)
    gray = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, estimate_t, r, eps)
    return t


def recover(im, t_estimate, a, sky):
    t_bound_down = 0.1
    t_bound_up = 0.9
    res = np.empty(im.shape, im.dtype)
    if RECOVER_REFINE == 1:
        res_sky = np.empty(im.shape, im.dtype)
        res_non_sky = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            res_sky[:, :, ind] = (im[:, :, ind]-a[ind])/cv2.max(t_estimate, t_bound_down) + a[ind]
            res_non_sky[:, :, ind] = (
                            im[:, :, ind] - a[ind]) / cv2.min(cv2.max(t_estimate, t_bound_down), t_bound_up) + a[ind]
        res[sky > 0] = res_sky[sky > 0]
        res[sky == 0] = res_non_sky[sky == 0]
    else:
        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - a[ind]) / cv2.max(t_estimate, t_bound_down) + a[ind]
    res[res > 255] = 255
    res[res < 0] = 0
    if RECOVER_WHITENING == 1:
        res_255 = res/255
        d_max = 100
        constant_b = 0.85
        res_tmp = np.zeros(im.shape, im.dtype)
        [h, w] = im.shape[:2]
        img_size = h * w

        for ind in range(3):
            base = res_255[:, :, ind]/max(res_255[:, :, ind].reshape(img_size))
            e = math.log(constant_b)/math.log(0.5)
            p = np.float_power(base, e)
            res_tmp[:, :, ind] = \
                (d_max*0.01/math.log10(max(res_255[:, :, ind].reshape(img_size))+1)) * \
                ((np.log(res_255[:, :, ind]+1))/np.log(2+(p*8)))
        res_tmp[res_tmp < 0] = 0
        res_tmp[res_tmp > 1] = 1
        res_tmp = res_tmp*255
        res_final = np.zeros(res_tmp.shape)
        res_final[sky > 0] = res_tmp[sky > 0]
        res_final[sky == 0] = res[sky == 0]
        return res_tmp
    else:
        return res


def train_model(x_model_1, w_model_1, x_model_2, w_model_2, x_model_3, w_model_3, bias):
    return np.multiply(x_model_1, w_model_1) + np.multiply(x_model_2, w_model_2) \
           + np.multiply(x_model_3, w_model_3) + bias


# Training Function starts here
def propagate(w1, w2, w3, b, x, y, t_refined, a_light, learning_rate):
    m, n = x.shape[0], x.shape[1]
    x1 = np.zeros(x.shape)
    x1[:, :, 0] = np.divide(x[:, :, 0], t_refined)
    x1[:, :, 1] = np.divide(x[:, :, 1], t_refined)
    x1[:, :, 2] = np.divide(x[:, :, 2], t_refined)
    x2 = np.zeros(x.shape)
    x2[:, :, 0] = np.divide(a_light[0], t_refined)
    x2[:, :, 1] = np.divide(a_light[1], t_refined)
    x2[:, :, 2] = np.divide(a_light[2], t_refined)

    # Forward propagation
    y0 = train_model(x1[:, :, 0], w1[0], x2[:, :, 0], w2[0], a_light[0], w3[0], b[0])
    y1 = train_model(x1[:, :, 1], w1[1], x2[:, :, 1], w2[1], a_light[1], w3[1], b[1])
    y2 = train_model(x1[:, :, 2], w1[2], x2[:, :, 2], w2[2], a_light[2], w3[2], b[2])

    # Backward propagation
    dw10 = np.sum(np.multiply(x1[:, :, 0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw11 = np.sum(np.multiply(x1[:, :, 1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw12 = np.sum(np.multiply(x1[:, :, 2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw1 = [dw10, dw11, dw12]

    dw20 = np.sum(np.multiply(x2[:, :, 0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw21 = np.sum(np.multiply(x2[:, :, 1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw22 = np.sum(np.multiply(x2[:, :, 2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw2 = [dw20, dw21, dw22]

    dw30 = np.sum(np.multiply(a_light[0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw31 = np.sum(np.multiply(a_light[1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw32 = np.sum(np.multiply(a_light[2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw3 = [dw30, dw31, dw32]

    db0 = np.sum(y0 - y[:, :, 0], dtype=np.float64)/(m*n)
    db1 = np.sum(y1 - y[:, :, 1], dtype=np.float64)/(m*n)
    db2 = np.sum(y2 - y[:, :, 2], dtype=np.float64)/(m*n)
    db = [db0, db1, db2]

    w1 = w1 - np.multiply(learning_rate, dw1)
    w2 = w2 - np.multiply(learning_rate, dw2)
    w3 = w3 - np.multiply(learning_rate, dw3)
    b = b - np.multiply(learning_rate, db)

    return w1, w2, w3, b


# Main stream starts here
# DCP:
def dcp_dehaze(dcp_img, weight):
    if DETECT_SKY == 1:
        dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(dcp_img, dtype=np.uint8))
        if sky_detected is None:
            [h, w] = dcp_img.shape[:2]
            sky = np.zeros([h, w])
        else:
            sky = sky_detected[:, :, 2]
            sky[sky != 255] = 1
            sky[sky == 255] = 0
    else:
        [h, w] = dcp_img.shape[:2]
        sky = np.zeros([h, w])
    a = atm_light(dcp_img, sky)
    t_estimated = transmission_estimate(dcp_img, a, sky, weight)
    if GUIDED_FILTER == 1:
        t_refined = transmission_refine(dcp_img, t_estimated)
    else:
        t_refined = t_estimated
    recovered_img = recover(dcp_img, t_refined, a, sky)
    return recovered_img


def train_method(train_imgs, train_refs):
    samples_per_fold = int(len(train_imgs) / FOLDS)

    ssim_folds = []
    psnr_folds = []
    test_num = 0
    test_recovered_imgs = []

    for fold_i in range(FOLDS):
        print('start at fold %d' % fold_i)
        test_range_up = (fold_i + 1) * samples_per_fold
        test_range_down = fold_i * samples_per_fold

        # training phase
        w1 = [0, 0, 0]
        w2 = [0, 0, 0]
        w3 = [0, 0, 0]
        b = [0, 0, 0]
        for it in range(ITER_NUM):
            for train_j in range(len(train_imgs)):
                if (train_j < test_range_down) or (train_j >= test_range_up):
                    train_img = train_imgs[train_j]
                    train_ref = train_refs[train_j]
                    [h, w] = train_img.shape[:2]
                    if DETECT_SKY == 1:
                        dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(train_img, dtype=np.uint8))
                        if sky_detected is None:
                            sky = np.zeros([h, w])
                        else:
                            sky = sky_detected[:, :, 2]
                            sky[sky != 255] = 1
                            sky[sky == 255] = 0
                    else:
                        sky = np.zeros([h, w])
                    a = atm_light(train_img, sky)
                    t_estimated = transmission_estimate(train_img, a, sky, None)
                    if GUIDED_FILTER == 1:
                        t_refined = transmission_refine(train_img, t_estimated)
                    else:
                        t_refined = t_estimated
                    t_refined[t_refined < 0.1] = 0.1
                    train_img_normalize = train_img / 255
                    train_ref_normalize = train_ref / 255
                    a_normalize = a / 255
                    w1, w2, w3, b = propagate(
                        w1, w2, w3, b, train_img_normalize, train_ref_normalize, t_refined, a_normalize,
                        learning_rate=0.05)

        # testing phase
        for test_j in range(len(train_imgs)):
            if (test_j >= test_range_down) and (test_j < test_range_up):
                test_img = train_imgs[test_j]
                test_ref = train_refs[test_j]
                [h, w] = test_img.shape[:2]
                if DETECT_SKY == 1:
                    dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(test_img, dtype=np.uint8))
                    if sky_detected is None:
                        sky = np.zeros([h, w])
                    else:
                        sky = sky_detected[:, :, 2]
                        sky[sky != 255] = 1
                        sky[sky == 255] = 0
                else:
                    sky = np.zeros([h, w])
                a = atm_light(test_img, sky)
                t_estimated = transmission_estimate(test_img, a, sky, None)
                if GUIDED_FILTER == 1:
                    t_refined = transmission_refine(test_img, t_estimated)
                else:
                    t_refined = t_estimated
                t_refined[t_refined < 0.1] = 0.1
                test_img_normalize = test_img/255
                a_normalize = a/255
                image_recovered = np.zeros(test_img.shape)
                image_recovered[:, :, 0] = train_model(
                    np.divide(test_img_normalize[:, :, 0], t_refined), w1[0],
                    np.divide(a_normalize[0], t_refined), w2[0], a_normalize[0], w3[0], b[0])
                image_recovered[:, :, 1] = train_model(
                    np.divide(test_img_normalize[:, :, 1], t_refined), w1[1],
                    np.divide(a_normalize[1], t_refined), w2[1], a_normalize[1], w3[1], b[1])
                image_recovered[:, :, 2] = train_model(
                    np.divide(test_img_normalize[:, :, 2], t_refined), w1[2],
                    np.divide(a_normalize[2], t_refined), w2[2], a_normalize[2], w3[2], b[2])
                test_ref_normalize = test_ref/255
                image_recovered[image_recovered > 1] = 1
                ssim_val = ssim(test_ref_normalize, image_recovered, multichannel=True)
                psnr_val = psnr(test_ref_normalize, image_recovered)
                ssim_folds.append(ssim_val)
                psnr_folds.append(psnr_val)
                test_recovered_imgs.append(image_recovered)
                test_num += 1

    print('%d images are tested.\n' % test_num)

    return psnr_folds, ssim_folds, test_recovered_imgs


def dcp_method(dcp_imgs, dcp_refs, files):
    dcp_ssims = []
    dcp_psnrs = []
    re_imgs = []
    test_num = 0
    if TRAIN_METHOD == 0:
        for d_j in range(len(dcp_imgs)):
            print('in process file: %s' % files[d_j])
            recovered_img = dcp_dehaze(dcp_imgs[d_j], None)
            dcp_ref_img = dcp_refs[d_j]
            ssim_val = ssim(dcp_ref_img, recovered_img, data_range=255, multichannel=True)
            psnr_val = psnr(dcp_ref_img, recovered_img, data_range=255)
            dcp_ssims.append(ssim_val)
            dcp_psnrs.append(psnr_val)
            re_imgs.append(recovered_img)
            test_num += 1
        print('%d images are tested.\n' % test_num)
    else:
        dcp_psnrs, dcp_ssims, re_imgs = train_method(dcp_imgs, dcp_refs)
    return dcp_psnrs, dcp_ssims, re_imgs


def dcp_test(dcp_imgs, dcp_refs):
    dcp_ssims = []
    dcp_psnrs = []
    dcp_weights_psnrs = []
    dcp_weights_ssims = []
    for dj in range(len(dcp_imgs)):
        print("test images at %d" % dj)
        ssims_per_weight = []
        psnrs_per_weight = []
        weights_record = []
        for di in range(100):
            weights = [0.5 + int(di / 10) * 0.05, 0.5 + (di % 10) * 0.05]
            recovered_img = dcp_dehaze(dcp_imgs[dj], weights)
            dcp_ref_img = dcp_refs[dj]
            ssim_val = ssim(dcp_ref_img, recovered_img, data_range=255, multichannel=True)
            psnr_val = psnr(dcp_ref_img, recovered_img, data_range=255)
            ssims_per_weight.append(ssim_val)
            psnrs_per_weight.append(psnr_val)
            weights_record.append(weights)
        dcp_ssims.append(np.max(ssims_per_weight))
        dcp_psnrs.append(np.max(psnrs_per_weight))
        index_1 = np.argmax(psnrs_per_weight)
        dcp_weights_psnrs.append(weights_record[int(index_1)])
        index_2 = np.argmax(ssims_per_weight)
        dcp_weights_ssims.append(weights_record[int(index_2)])
    file_weight = open('results.txt', 'w+')
    for j_weight in range(len(dcp_ssims)):
        file_weight.write("%f\t" % dcp_psnrs[j_weight])
        file_weight.write("%f\t" % dcp_ssims[j_weight])
        file_weight.write("%s\t" % dcp_weights_psnrs[j_weight])
        file_weight.write("%s\n" % dcp_weights_ssims[j_weight])
    file_weight.close()
    return


def print_image_information(print_imgs):
    brightness_sky = []
    brightness_non_sky = []
    a_light = []
    is_sky = []
    for j_img in range(len(print_imgs)):
        print_img = print_imgs[j_img]
        [h, w] = print_img.shape[:2]
        img_size = h*w
        dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(print_img, dtype=np.uint8))
        if sky_detected is None:
            sky = np.zeros([h, w])
            is_sky.append(0)
        else:
            sky = sky_detected[:, :, 2]
            sky[sky != 255] = 1
            sky[sky == 255] = 0
            is_sky.append(1)
        a = atm_light(print_img, sky)
        a_light.append(a)
        bright_sky = np.zeros(print_img.shape, print_img.dtype)
        bright_non_sky = np.zeros(print_img.shape, print_img.dtype)
        bright_sky[sky > 0] = print_img[sky > 0]
        bright_non_sky[sky == 0] = print_img[sky == 0]
        brightness_sky.append(np.max(cv2.cvtColor(np.array(bright_sky, dtype=np.uint8),
                                                  cv2.COLOR_BGR2GRAY).reshape(img_size)))
        brightness_non_sky.append(
            np.max(cv2.cvtColor(np.array(bright_non_sky, dtype=np.uint8), cv2.COLOR_BGR2GRAY).reshape(img_size)))

    file_print = open('image_information.txt', 'w+')
    for j_print in range(len(a_light)):
        file_print.write("%d\t" % is_sky[j_print])
        file_print.write("%f\t" % brightness_sky[j_print])
        file_print.write("%f\t" % brightness_non_sky[j_print])
        file_print.write("%s\n" % a_light[j_print])
    file_print.close()
    return


def test_sky(im):
    test_bopt, final = Sky_Detection.detect_sky(im)
    if final is not None:
        b, g, sky = cv2.split(final)
        final[sky == 255] = [0, 0, 0]
    return test_bopt, final


if __name__ == '__main__':
    data_dir = 'SOTS'
    in_or_out = 'outdoor'
    im_dir = os.path.join(data_dir, in_or_out, 'hazy')
    ref_dir = os.path.join(data_dir, in_or_out, 'gt')
    result_dir = os.path.join(data_dir, in_or_out, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir(im_dir):
        print ('Given path %s not found' % format(im_dir))
        sys.exit(-1)

    if not os.path.isdir(ref_dir):
        print ('Given path %s not found' % format(ref_dir))
        sys.exit(-1)

    img_all = []
    ref_img_all = []
    file_names = []

    for fn in os.listdir(im_dir):
        if fn[0] == '.':
            continue
        im_path = os.path.join(im_dir, fn)
        fn_ref = fn[:4] + '.png'
        ref_path = os.path.join(ref_dir, fn_ref)
        assert (os.path.exists(im_path)), 'Annotation: %s does not exist' % format(im_path)
        assert (os.path.exists(ref_path)), 'Annotation: %s does not exist' % format(ref_path)
        ref_src = cv2.imread(ref_path)
        ref_img = ref_src.astype('float64')
        src = cv2.imread(im_path)
        img = src.astype('float64')
        img_all.append(img)
        ref_img_all.append(ref_img)
        file_names.append(fn_ref)
    print('%d images are loaded.' % len(img_all))

    if METHOD == 1:
        print('dehaze using DCP')
        psnrs, ssims, recovered_imgs = dcp_method(img_all, ref_img_all, file_names)
        print('The average PSNR value is %0.4f.\n' % np.mean(psnrs))
        print('The average SSIM value is %0.4f.\n' % np.mean(ssims))
        if ENABLE_PRINT_FINAL_IMAGE > 0:
            for i in range(len(recovered_imgs)):
                cv2.imwrite(os.path.join(result_dir, file_names[i]), recovered_imgs[i])
            file_result = open(os.path.join(result_dir, 'results.txt'), 'w+')
            for j in range(len(ssims)):
                file_result.write("%s\t" % file_names[j])
                file_result.write("%f\t" % psnrs[j])
                file_result.write("%f\n" % ssims[j])
            file_result.close()
    elif METHOD == 2:
        bopt, img_recovered = test_sky(img_all[0])
        img_test = img_all[0]
        ref_img = ref_img_all[0]
        if img_recovered is not None:
            cv2.imshow('Original Image', img_all[0])
            cv2.imshow('Recovered Image', img_recovered)
            cv2.imshow('Reference Image', ref_img_all[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif METHOD == 3:
        dcp_test(img_all, ref_img_all)
    elif METHOD == 4:
        print_image_information(img_all)
    else:
        assert()

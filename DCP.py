import cv2
import math
import numpy as np


# Parameters defined here
LARGE_PATCH_SIZE = 15
SMALL_PATCH_SIZE = 3
HAZE_WEIGHT = 0.95
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
PIXEL_MAX = 255.0

# Function define here
PRIOR_OPTIMIZE = 1


def dark_channel_prior(input_img, sky):
    dc = dark_channel(input_img)
    [h, w] = input_img.shape[:2]
    if PRIOR_OPTIMIZE == 1 and np.sum(sky.reshape(h*w)) > h*w*0.05:
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


def atm_light_brightest_in_dc(im, sky):
    img_sky = im.copy()
    img_sky[sky == 0] = [0, 0, 0]
    img_non_sky = im.copy()
    [h, w] = im.shape[:2]
    img_size = h * w
    if sum(sky.reshape(img_size)) > img_size*0.05:
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
    return a


def atm_light_average(im, sky):
    img_sky = im.copy()
    img_sky[sky == 0] = [0, 0, 0]
    img_non_sky = im.copy()
    [h, w] = im.shape[:2]
    img_size = h * w
    if sum(sky.reshape(img_size)) > img_size*0.05:
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
        x0 = 0
        y0 = 0
        x1 = h - 1
        y1 = w - 1
        while (x1 - x0) * (y1 - y0) > 0.05 * img_size:
            x1 = int((x1 + x0) / 2)
            y1 = int((y1 + y0) / 2)
            x0, x1, y0, y1 = sub_block(im, x0, x1, y0, y1)
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
        block_value.append(sub_block_value(im[block_all[block_i][0]:block_all[block_i][1],
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


def transmission_estimate(im, a, sky):
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind]/a[ind]
    transmission = 1 - HAZE_WEIGHT * dark_channel_prior(im3, sky)
    return transmission


def transmission_estimate_inverse_image(im, a, sky):
    im3 = np.empty(im.shape, im.dtype)
    [h, w] = im.shape[:2]
    img_size = h * w
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind]/a[ind]
    if sum(sky.reshape(img_size)) > h * w * 0.05:
        transmission_non_sky = 1 - HAZE_WEIGHT * dark_channel_prior(im3, sky)
        img_inverse = 255 - im
        sky_inverse = sky.copy()
        sky_inverse += 1
        sky_inverse[sky_inverse > 1] = 0
        a_inverse = atm_light_brightest_in_dc(img_inverse, sky_inverse)
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
    res_sky = np.empty(im.shape, im.dtype)
    res_non_sky = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        res_sky[:, :, ind] = (im[:, :, ind] - a[ind]) / cv2.max(t_estimate, t_bound_down) + a[ind]
        res_non_sky[:, :, ind] = (
                                         im[:, :, ind] - a[ind]) / cv2.min(cv2.max(t_estimate, t_bound_down),
                                                                           t_bound_up) + a[ind]
    res[sky > 0] = res_sky[sky > 0]
    res[sky == 0] = res_non_sky[sky == 0]
    res[res > 255] = 255
    res[res < 0] = 0
    return res


def recover_whiten(im, res, sky):
    res_255 = res / 255
    d_max = 100
    constant_b = 0.85
    res_tmp = np.zeros(im.shape, im.dtype)
    [h, w] = im.shape[:2]
    img_size = h * w

    for ind in range(3):
        base = res_255[:, :, ind] / max(res_255[:, :, ind].reshape(img_size))
        e = math.log(constant_b) / math.log(0.5)
        p = np.float_power(base, e)
        res_tmp[:, :, ind] = \
            (d_max * 0.01 / math.log10(max(res_255[:, :, ind].reshape(img_size)) + 1)) * \
            ((np.log(res_255[:, :, ind] + 1)) / np.log(2 + (p * 8)))
    res_tmp[res_tmp < 0] = 0
    res_tmp[res_tmp > 1] = 1
    res_tmp = res_tmp * 255
    res_final = np.zeros(res_tmp.shape)
    res_final[sky > 0] = res_tmp[sky > 0]
    res_final[sky == 0] = res[sky == 0]
    return res_tmp

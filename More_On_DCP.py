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
HAZE_WEIGHT = 0.9
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
PIXEL_MAX = 255.0

# Function define here
DETECT_SKY = 0
PRIOR_OPTIMIZE = 0
RECOVER_REFINE = 1
RECOVER_WHITENING = 1
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
ENABLE_PRINT_FINAL_IMAGE = 0


METHOD = 3  # 1: DCP; 2: test; 3: DCP Weight test

FOLDS = 5


# DCP starts here
def dark_channel_prior(input_img, sky):
    dc = dark_channel(input_img)
    if PRIOR_OPTIMIZE == 1:
        assert(DETECT_SKY > 0)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (LARGE_PATCH_SIZE, LARGE_PATCH_SIZE))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (SMALL_PATCH_SIZE, SMALL_PATCH_SIZE))
        dcp_large = cv2.erode(dc, kernel_large)
        dcp_small = cv2.erode(dc, kernel_small)
        [h, w] = input_img.shape[:2]
        dcp = np.zeros(input_img.shape[:2])
        for i in range(h):
            for j in range(w):
                x = i-2
                y = j-2
                if i < 2:
                    x = 0
                if j < 2:
                    y = 0
                if i >= h-6:
                    x = h-6
                if j >= w-6:
                    y = w-6
                region = sky[x:x+5, y:y+5]
                sum_region = sum(region.reshape(25))
                if 8 < sum_region < 18:
                    dcp[i, j] = dcp_small[i, j]
                else:
                    dcp[i, j] = dcp_large[i, j]
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
    in_img = im.copy()
    [h, w] = in_img.shape[:2]
    img_size = h * w

    if BRIGHTEST_IN_DC:
        dc = dark_channel_prior(in_img, sky)
        num_pixel = int(max(math.floor(img_size * BRIGHTEST_PIXELS_PERCENTAGE), 1))
        dark_channel_vec = dc.reshape(img_size)
        img_vec = in_img.reshape(img_size, 3)
        indices = (-dark_channel_vec).argsort()[:num_pixel]
        brightest_pixel = img_vec[indices]
        b = brightest_pixel[:, 0]
        g = brightest_pixel[:, 1]
        r = brightest_pixel[:, 2]
        brightest = np.argmax(np.power(b, 2) + np.power(g, 2) + np.power(r, 2))
        a = brightest_pixel[brightest]
    elif sum(sky.reshape(img_size)) > img_size*0.05:
        in_img[sky == 0] = [0, 0, 0]
        num_pixel = int(max(math.floor(img_size * BRIGHTEST_PIXELS_PERCENTAGE), 1))
        b = in_img[:, :, 0].reshape(img_size)
        g = in_img[:, :, 1].reshape(img_size)
        r = in_img[:, :, 2].reshape(img_size)

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
            x0, x1, y0, y1 = sub_block(img, x0, x1, y0, y1)
        block = in_img[x0:x1, y0:y1]
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
    for i in range(4):
        block_value.append(sub_block_value(img[block_all[i][0]:block_all[i][1], block_all[i][2]:block_all[i][3]]))
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

    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind]/a[ind]

    if weight is None:
        transmission = 1 - HAZE_WEIGHT*dark_channel_prior(im3, sky)
    else:
        transmission_sky = 1 - weight[0] * dark_channel_prior(im3, sky)
        transmission_non_sky = 1 - weight[0] * dark_channel_prior(im3, sky)
        transmission = transmission_sky
        transmission[sky == 0] = transmission_non_sky[sky == 0]
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
            res_non_sky[:, :, ind] = (im[:, :, ind] - a[ind]) / \
                             cv2.min(cv2.max(t_estimate, t_bound_down), t_bound_up) + a[ind]
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
            res_tmp[:, :, ind] = (d_max*0.01/math.log10(max(res_255[:, :, ind].reshape(img_size))+1)) * \
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


def dcp_method(dcp_imgs, dcp_refs, files):
    dcp_ssims = []
    dcp_psnrs = []
    re_imgs = []
    test_num = 0
    for j in range(len(dcp_imgs)):
        print('in process file: %s' % files[j])
        recovered_img = dcp_dehaze(dcp_imgs[j], None)
        dcp_ref_img = dcp_refs[j]
        ssim_val = ssim(dcp_ref_img, recovered_img, data_range=255, multichannel=True)
        psnr_val = psnr(dcp_ref_img, recovered_img, data_range=255)
        dcp_ssims.append(ssim_val)
        dcp_psnrs.append(psnr_val)
        re_imgs.append(recovered_img)
        test_num += 1
    print('%d images are tested.\n' % test_num)
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
        for di in range(100):
            weights = [0.5 + int(di / 10) * 0.05, 0.5 + (di % 10) * 0.05]
            recovered_img = dcp_dehaze(dcp_imgs[dj], weights)
            dcp_ref_img = dcp_refs[dj]
            ssim_val = ssim(dcp_ref_img, recovered_img, data_range=255, multichannel=True)
            psnr_val = psnr(dcp_ref_img, recovered_img, data_range=255)
            ssims_per_weight.append(ssim_val)
            psnrs_per_weight.append(psnr_val)
        dcp_ssims.append(np.max(ssims_per_weight))
        dcp_psnrs.append(np.max(psnrs_per_weight))
        index = np.argmax(psnrs_per_weight)
        dcp_weights_psnrs.append(weights[index])
        index = np.argmax(ssims_per_weight)
        dcp_weights_ssims.append(weights[index])
    file_weight = open('results.txt', 'w+')
    for j_weight in range(len(dcp_ssims)):
        file_weight.write("%f\t" % dcp_psnrs[j_weight])
        file_weight.write("%f\t" % dcp_ssims[j_weight])
        file_weight.write("%f\t" % dcp_weights_psnrs[j_weight])
        file_weight.write("%f\n" % dcp_weights_ssims[j_weight])
    file_result.close()
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
        break
    print('%d images are loaded.' % len(img_all))

    if METHOD == 1:
        print('dehaze using DCP')
        psnrs, ssims, recovered_imgs= dcp_method(img_all, ref_img_all, file_names)
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
    else:
        assert()

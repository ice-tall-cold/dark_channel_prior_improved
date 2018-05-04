#!/bin/env python
#
# This program has several way to improve the Dark Channel Prior method for image dehazing.
#
# Author: Wenrui Zhang & Binghan Li
# Date: Apr 1, 2018

import os
import sys
import cv2
import numpy as np
import Sky_Detection
import DCP
import logging
import Training_Method as TM
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


# Function define here
DETECT_SKY = 0
RECOVER_WHITENING = 0
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 0
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 2  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test

CROSS_VALIDATION = 1
FOLDS = 5
ITER_NUM = 3


# DCP:
def dcp_dehaze(dcp_img, sky):
    if BRIGHTEST_IN_DC == 1:
        a = DCP.atm_light_brightest_in_dc(dcp_img, sky)
    else:
        a = DCP.atm_light_average(dcp_img, sky)
    if INVERSE_SKY == 1:
        t_estimated = DCP.transmission_estimate_inverse_image(dcp_img, a, sky)
    else:
        t_estimated = DCP.transmission_estimate(dcp_img, a, sky)
    if GUIDED_FILTER == 1:
        t_refined = DCP.transmission_refine(dcp_img, t_estimated)
    else:
        t_refined = t_estimated
    recovered_tmp = DCP.recover(dcp_img, t_refined, a, sky)
    if RECOVER_WHITENING == 1:
        recovered_img = DCP.recover_whiten(dcp_img, recovered_tmp, sky)
    else:
        recovered_img = recovered_tmp
    return recovered_img


def train_method(test_imgs, test_refs, train_imgs, train_refs, files):
    test_skys = []
    count = 0
    for im_in in test_imgs:
        count += 1
        logging.info("sky segmentation at %d" % count)
        [h, w] = im_in.shape[:2]
        if DETECT_SKY == 1:
            dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(im_in, dtype=np.uint8))
            if sky_detected is None:
                sky = np.zeros([h, w])
            else:
                sky = sky_detected[:, :, 2]
                sky[sky != 255] = 1
                sky[sky == 255] = 0
        else:
            sky = np.zeros([h, w])
        test_skys.append(sky)

    ssim_folds = []
    psnr_folds = []
    test_num = 0
    test_recovered_imgs = []

    if CROSS_VALIDATION == 1:
        samples_per_fold = int(len(test_imgs) / FOLDS)
        for fold_i in range(FOLDS):
            logging.info('start at fold %d' % fold_i)
            test_range_up = (fold_i + 1) * samples_per_fold
            test_range_down = fold_i * samples_per_fold

            # training phase
            w1 = [0, 0, 0]
            w2 = [0, 0, 0]
            w3 = [0, 0, 0]
            b = [0, 0, 0]
            for it in range(ITER_NUM):
                logging.info('train at iteration %d' % it)
                for train_j in range(len(test_imgs)):
                    if (train_j < test_range_down) or (train_j >= test_range_up):
                        train_img = test_imgs[train_j]
                        train_ref = test_refs[train_j]
                        train_sky = test_skys[train_j]
                        if BRIGHTEST_IN_DC == 1:
                            a = DCP.atm_light_brightest_in_dc(train_img, train_sky)
                        else:
                            a = DCP.atm_light_average(train_img, train_sky)
                        if INVERSE_SKY == 1:
                            t_estimated = DCP.transmission_estimate_inverse_image(train_img, a, train_sky)
                        else:
                            t_estimated = DCP.transmission_estimate(train_img, a, train_sky)
                        if GUIDED_FILTER == 1:
                            t_refined = DCP.transmission_refine(train_img, t_estimated)
                        else:
                            t_refined = t_estimated
                        t_refined[t_refined < 0.1] = 0.1
                        train_img_normalize = train_img / 255
                        train_ref_normalize = train_ref / 255
                        a_normalize = a / 255
                        w1, w2, w3, b = TM.propagate(
                            w1, w2, w3, b, train_img_normalize, train_ref_normalize, t_refined, a_normalize,
                            learning_rate=0.23)

            # testing phase
            for test_j in range(len(test_imgs)):
                if (test_j >= test_range_down) and (test_j < test_range_up):
                    test_img = test_imgs[test_j]
                    test_ref = test_refs[test_j]
                    test_sky = test_skys[test_j]
                    logging.info('in process file: %s' % files[test_j])
                    if BRIGHTEST_IN_DC == 1:
                        a = DCP.atm_light_brightest_in_dc(test_img, test_sky)
                    else:
                        a = DCP.atm_light_average(test_img, test_sky)
                    if INVERSE_SKY == 1:
                        t_estimated = DCP.transmission_estimate_inverse_image(test_img, a, test_sky)
                    else:
                        t_estimated = DCP.transmission_estimate(test_img, a, test_sky)
                    if GUIDED_FILTER == 1:
                        t_refined = DCP.transmission_refine(test_img, t_estimated)
                    else:
                        t_refined = t_estimated
                    t_refined[t_refined < 0.1] = 0.1
                    test_img_normalize = test_img / 255
                    a_normalize = a / 255
                    image_recovered = np.zeros(test_img.shape)
                    image_recovered[:, :, 0] = TM.train_model(
                        np.divide(test_img_normalize[:, :, 0], t_refined), w1[0],
                        np.divide(a_normalize[0], t_refined), w2[0], a_normalize[0], w3[0], b[0])
                    image_recovered[:, :, 1] = TM.train_model(
                        np.divide(test_img_normalize[:, :, 1], t_refined), w1[1],
                        np.divide(a_normalize[1], t_refined), w2[1], a_normalize[1], w3[1], b[1])
                    image_recovered[:, :, 2] = TM.train_model(
                        np.divide(test_img_normalize[:, :, 2], t_refined), w1[2],
                        np.divide(a_normalize[2], t_refined), w2[2], a_normalize[2], w3[2], b[2])
                    test_ref_normalize = test_ref / 255
                    image_recovered[image_recovered > 1] = 1
                    ssim_val = ssim(test_ref_normalize, image_recovered, multichannel=True)
                    psnr_val = psnr(test_ref_normalize, image_recovered)
                    ssim_folds.append(ssim_val)
                    psnr_folds.append(psnr_val)
                    logging.info('SSIM: %f' % ssim_val)
                    logging.info('PSNR: %f' % psnr_val)
                    test_recovered_imgs.append(image_recovered)
                    test_num += 1

    else:
        train_skys = []
        count = 0
        for im_in in train_imgs:
            count += 1
            logging.info("sky segmentation at %d" % count)
            [h, w] = im_in.shape[:2]
            if DETECT_SKY == 1:
                dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(im_in, dtype=np.uint8))
                if sky_detected is None:
                    sky = np.zeros([h, w])
                else:
                    sky = sky_detected[:, :, 2]
                    sky[sky != 255] = 1
                    sky[sky == 255] = 0
            else:
                sky = np.zeros([h, w])
            train_skys.append(sky)

        # training phase
        w1 = [0, 0, 0]
        w2 = [0, 0, 0]
        w3 = [0, 0, 0]
        b = [0, 0, 0]
        for it in range(ITER_NUM):
            logging.info('train at iteration %d' % it)
            for train_j in range(len(train_imgs)):
                train_img = train_imgs[train_j]
                train_ref = train_refs[train_j]
                train_sky = train_skys[train_j]
                if BRIGHTEST_IN_DC == 1:
                    a = DCP.atm_light_brightest_in_dc(train_img, train_sky)
                else:
                    a = DCP.atm_light_average(train_img, train_sky)
                if INVERSE_SKY == 1:
                    t_estimated = DCP.transmission_estimate_inverse_image(train_img, a, train_sky)
                else:
                    t_estimated = DCP.transmission_estimate(train_img, a, train_sky)
                if GUIDED_FILTER == 1:
                    t_refined = DCP.transmission_refine(train_img, t_estimated)
                else:
                    t_refined = t_estimated
                t_refined[t_refined < 0.1] = 0.1
                train_img_normalize = train_img / 255
                train_ref_normalize = train_ref / 255
                a_normalize = a / 255
                w1, w2, w3, b = TM.propagate(
                    w1, w2, w3, b, train_img_normalize, train_ref_normalize, t_refined, a_normalize,
                    learning_rate=0.23)

        # testing phase
        for test_j in range(len(test_imgs)):
            test_img = test_imgs[test_j]
            test_ref = test_refs[test_j]
            test_sky = test_skys[test_j]
            logging.info('in process file: %s' % files[test_j])
            if BRIGHTEST_IN_DC == 1:
                a = DCP.atm_light_brightest_in_dc(test_img, test_sky)
            else:
                a = DCP.atm_light_average(test_img, test_sky)
            if INVERSE_SKY == 1:
                t_estimated = DCP.transmission_estimate_inverse_image(test_img, a, test_sky)
            else:
                t_estimated = DCP.transmission_estimate(test_img, a, test_sky)
            if GUIDED_FILTER == 1:
                t_refined = DCP.transmission_refine(test_img, t_estimated)
            else:
                t_refined = t_estimated
            t_refined[t_refined < 0.1] = 0.1
            test_img_normalize = test_img / 255
            a_normalize = a / 255
            image_recovered = np.zeros(test_img.shape)
            image_recovered[:, :, 0] = TM.train_model(
                np.divide(test_img_normalize[:, :, 0], t_refined), w1[0],
                np.divide(a_normalize[0], t_refined), w2[0], a_normalize[0], w3[0], b[0])
            image_recovered[:, :, 1] = TM.train_model(
                np.divide(test_img_normalize[:, :, 1], t_refined), w1[1],
                np.divide(a_normalize[1], t_refined), w2[1], a_normalize[1], w3[1], b[1])
            image_recovered[:, :, 2] = TM.train_model(
                np.divide(test_img_normalize[:, :, 2], t_refined), w1[2],
                np.divide(a_normalize[2], t_refined), w2[2], a_normalize[2], w3[2], b[2])
            test_ref_normalize = test_ref / 255
            image_recovered[image_recovered > 1] = 1
            ssim_val = ssim(test_ref_normalize, image_recovered, multichannel=True)
            psnr_val = psnr(test_ref_normalize, image_recovered)
            ssim_folds.append(ssim_val)
            psnr_folds.append(psnr_val)
            logging.info('SSIM: %f' % ssim_val)
            logging.info('PSNR: %f' % psnr_val)
            test_recovered_imgs.append(image_recovered)
            test_num += 1

    logging.info('%d images are tested.\n' % test_num)

    return psnr_folds, ssim_folds, test_recovered_imgs


def dcp_method(dcp_imgs, dcp_refs, files):
    count = 0
    skys = []
    for im_in in dcp_imgs:
        count += 1
        logging.info("sky segmentation at %d" % count)
        [h, w] = im_in.shape[:2]
        if DETECT_SKY == 1:
            dcp_bopt, sky_detected = Sky_Detection.detect_sky(np.array(im_in, dtype=np.uint8))
            if sky_detected is None:
                sky = np.zeros([h, w])
            else:
                sky = sky_detected[:, :, 2]
                sky[sky != 255] = 1
                sky[sky == 255] = 0
        else:
            sky = np.zeros([h, w])
        skys.append(sky)
    logging.info('sky processed')
    dcp_ssims = []
    dcp_psnrs = []
    re_imgs = []
    test_num = 0

    for d_j in range(len(dcp_imgs)):
        logging.info('in process file: %s' % files[d_j])
        recovered_img = dcp_dehaze(dcp_imgs[d_j], skys[d_j])
        dcp_ref_img = dcp_refs[d_j]
        ssim_val = ssim(dcp_ref_img, recovered_img, data_range=255, multichannel=True)
        psnr_val = psnr(dcp_ref_img, recovered_img, data_range=255)
        dcp_ssims.append(ssim_val)
        dcp_psnrs.append(psnr_val)
        logging.info('SSIM: %f' % ssim_val)
        logging.info('PSNR: %f' % psnr_val)
        re_imgs.append(recovered_img)
        test_num += 1
    logging.info('%d images are tested.\n' % test_num)
    return dcp_psnrs, dcp_ssims, re_imgs


def sky_segmentation_test(ims):
    segments = []
    for im in ims:
        test_bopt, final = Sky_Detection.detect_sky(im)
        if final is not None:
            b, g, sky = cv2.split(final)
            final[sky == 255] = [0, 0, 0]
        segments.append(final)
    return segments


if __name__ == '__main__':
    try:
        os.remove('main.log')
    except OSError:
        pass
    logging.basicConfig(filename='main.log', level=logging.DEBUG)
    data_dir = 'SOTS'
    in_door = 'indoor'
    out_door = 'outdoor'
    im_dir_in = os.path.join(data_dir, in_door, 'hazy')
    ref_dir_in = os.path.join(data_dir, in_door, 'gt')
    im_dir_out = os.path.join(data_dir, out_door, 'hazy')
    ref_dir_out = os.path.join(data_dir, out_door, 'gt')
    result_dcp_dir = os.path.join(data_dir, 'results_dcp')
    result_train_dir = os.path.join(data_dir, 'results_train')
    result_sky_dir = os.path.join(data_dir, 'results_sky')
    im_dir_ots = 'OTS/haze'
    ref_dir_ots = 'OTS/gt'
    if not os.path.exists(result_dcp_dir):
        os.makedirs(result_dcp_dir)
    if not os.path.exists(result_train_dir):
        os.makedirs(result_train_dir)
    if not os.path.exists(result_sky_dir):
        os.makedirs(result_sky_dir)
    if not os.path.isdir(im_dir_in):
        logging.error('Given path %s not found' % format(im_dir_in))
        sys.exit(-1)
    if not os.path.isdir(im_dir_out):
        logging.error('Given path %s not found' % format(im_dir_out))
        sys.exit(-1)
    if not os.path.isdir(ref_dir_in):
        logging.error('Given path %s not found' % format(ref_dir_in))
        sys.exit(-1)
    if not os.path.isdir(ref_dir_out):
        logging.error('Given path %s not found' % format(ref_dir_out))
        sys.exit(-1)

    img_all = []
    ref_img_all = []
    file_names = []
    train_all = []
    train_ref_all = []

    if CROSS_VALIDATION == 0:
        if not os.path.isdir(im_dir_ots):
            logging.error('Given path %s not found' % format(im_dir_ots))
            sys.exit(-1)
        if not os.path.isdir(ref_dir_ots):
            logging.error('Given path %s not found' % format(ref_dir_ots))
            sys.exit(-1)
        for fn in os.listdir(im_dir_ots):
            if fn[0] == '.':
                continue
            im_path = os.path.join(im_dir_ots, fn)
            fn_ref = fn[:4] + '.png'
            ref_path = os.path.join(ref_dir_ots, fn_ref)
            assert (os.path.exists(im_path)), 'Annotation: %s does not exist' % format(im_path)
            assert (os.path.exists(ref_path)), 'Annotation: %s does not exist' % format(ref_path)
            ref_img = cv2.imread(ref_path).astype('float64')
            img = cv2.imread(im_path).astype('float64')
            train_all.append(img)
            train_ref_all.append(ref_img)

    for fn in os.listdir(im_dir_in):
        if fn[0] == '.':
            continue
        im_path = os.path.join(im_dir_in, fn)
        fn_ref = fn[:4] + '.png'
        ref_path = os.path.join(ref_dir_in, fn_ref)
        assert (os.path.exists(im_path)), 'Annotation: %s does not exist' % format(im_path)
        assert (os.path.exists(ref_path)), 'Annotation: %s does not exist' % format(ref_path)
        ref_img = cv2.imread(ref_path).astype('float64')
        img = cv2.imread(im_path).astype('float64')
        img_all.append(img)
        ref_img_all.append(ref_img)
        file_names.append(fn_ref)

    for fn in os.listdir(im_dir_out):
        if fn[0] == '.':
            continue
        im_path = os.path.join(im_dir_out, fn)
        fn_ref = fn[:4] + '.png'
        ref_path = os.path.join(ref_dir_out, fn_ref)
        assert (os.path.exists(im_path)), 'Annotation: %s does not exist' % format(im_path)
        assert (os.path.exists(ref_path)), 'Annotation: %s does not exist' % format(ref_path)
        ref_img = cv2.imread(ref_path).astype('float64')
        img = cv2.imread(im_path).astype('float64')
        img_all.append(img)
        ref_img_all.append(ref_img)
        file_names.append(fn_ref)
    logging.info('%d images are loaded.' % len(img_all))

    if METHOD == 1:
        logging.info('dehaze using DCP')
        psnrs, ssims, recovered_imgs = dcp_method(img_all, ref_img_all, file_names)
        logging.info('The average PSNR value is %0.4f.\n' % np.mean(psnrs))
        logging.info('The average SSIM value is %0.4f.\n' % np.mean(ssims))
        if ENABLE_PRINT_FINAL_IMAGE > 0:
            for i in range(len(recovered_imgs)):
                cv2.imwrite(os.path.join(result_dcp_dir, file_names[i]), recovered_imgs[i])
            file_result = open('results_dcp.txt', 'w+')
            for j in range(len(ssims)):
                file_result.write("%s\t" % file_names[j])
                file_result.write("%f\t" % psnrs[j])
                file_result.write("%f\n" % ssims[j])
            file_result.close()
    elif METHOD == 2:
        logging.info('dehaze using Train method')
        psnrs, ssims, recovered_imgs = train_method(img_all, ref_img_all, train_all, train_ref_all, file_names)
        logging.info('The average PSNR value is %0.4f.\n' % np.mean(psnrs))
        logging.info('The average SSIM value is %0.4f.\n' % np.mean(ssims))
        if ENABLE_PRINT_FINAL_IMAGE > 0:
            for i in range(len(recovered_imgs)):
                cv2.imwrite(os.path.join(result_train_dir, file_names[i]), recovered_imgs[i])
            file_result = open('results_train_method.txt', 'w+')
            for j in range(len(ssims)):
                file_result.write("%s\t" % file_names[j])
                file_result.write("%f\t" % psnrs[j])
                file_result.write("%f\n" % ssims[j])
            file_result.close()
    elif METHOD == 3:
        logging.info('show sky segmentation results')
        recovered_imgs = sky_segmentation_test(img_all)
        if ENABLE_PRINT_FINAL_IMAGE > 0:
            for i in range(len(recovered_imgs)):
                cv2.imwrite(os.path.join(result_sky_dir, file_names[i]), recovered_imgs[i])
    else:
        assert()

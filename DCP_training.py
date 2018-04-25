import os
import sys
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

PATCH_SIZE = 3
HAZE_WEIGHT = 0.95
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
PIXEL_MAX = 255.0

FOLDS = 5

METHOD = 2  # 1: learning method; 2: DCP


# learning method start here
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_prime(z):
    s = sigmoid(z) * (1 - sigmoid(z))
    return s


def propagate(w, b, x, y, learning_rate):
    m, n = x.shape[0], x.shape[1]

    # Forward propagation
    a0 = (np.multiply(w[0], x[:, :, 0]) + b[0])
    a1 = (np.multiply(w[1], x[:, :, 1]) + b[1])
    a2 = (np.multiply(w[2], x[:, :, 2]) + b[2])
    # cost = (np.sum(np.multiply(Y[:,:,0],np.log(A0)) + np.multiply((np.ones((m,n)) - Y[:,:,0]), np.log(np.ones((m,n)) - A0)))+
    #      np.sum(np.multiply(Y[:,:,1],np.log(A1)) + np.multiply((np.ones((m,n)) - Y[:,:,1]), np.log(np.ones((m,n)) - A1)))+
    #       np.sum(np.multiply(Y[:,:,2],np.log(A2)) + np.multiply((np.ones((m,n)) - Y[:,:,2]), np.log(np.ones((m,n)) - A2))))*(-1/(m*n*3))

    # Backward propagation
    dw0 = np.sum(np.multiply((x[:, :, 0]), (a0 - y[:, :, 0])), dtype=np.float64) / (m * n)
    dw1 = np.sum(np.multiply((x[:, :, 1]), (a1 - y[:, :, 1])), dtype=np.float64) / (m * n)
    dw2 = np.sum(np.multiply((x[:, :, 2]), (a2 - y[:, :, 2])), dtype=np.float64) / (m * n)
    # dw0 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[0], X[:,:,0])), (A0 - Y[:,:,0])), dtype = np.float64)/(m*n)
    # dw1 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[1], X[:,:,1])), (A0 - Y[:,:,1])), dtype = np.float64)/(m*n)
    # dw2 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[2], X[:,:,2])), (A0 - Y[:,:,2])), dtype = np.float64)/(m*n)
    db0 = np.sum(a0 - y[:, :, 0], dtype=np.float64) / (m * n)
    db1 = np.sum(a1 - y[:, :, 1], dtype=np.float64) / (m * n)
    db2 = np.sum(a2 - y[:, :, 2], dtype=np.float64) / (m * n)

    # cost = np.squeeze(cost)
    dw = [dw0, dw1, dw2]
    db = [db0, db1, db2]
    w -= np.multiply(learning_rate, dw)
    b -= np.multiply(learning_rate, db)

    return w, b


# DCP starts here
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
    a = np.zeros(3)
    a[0] = max(brightest_r)
    a[1] = max(brightest_g)
    a[2] = max(brightest_b)

# average form of a
#    atm_sum = np.zeros(3)
#    for ind in range(1, num_pixel):
#       atm_sum = atm_sum + img_vec[indices[ind]]
#    a = atm_sum / num_pixel

    return a


def transmission_estimate(im, a, sz):
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind]/a[ind]

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


def recover(im, t_estimate, a, t_bound=0.1):
    res = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-a[ind])/cv2.max(t_estimate, t_bound) + a[ind]
    return res


# Main stream starts here
# Training method
def train_method(imgs, refs):
    samples_per_fold = int(len(imgs) / FOLDS)

    ssim_folds = []
    psnr_folds = []
    test_num = 0

    for i in range(FOLDS):
        print('start at fold %d' % i)
        test_range_up = (i + 1) * samples_per_fold
        test_range_down = i * samples_per_fold

        # training phase
        w = [0, 0, 0]
        b = [0, 0, 0]
        sum_ssim = 0
        sum_psnr = 0
        for j in range(len(imgs)):
            if (j < test_range_down) or (j >= test_range_up):
                w, b = propagate(w, b, imgs[j], refs[j], learning_rate=0.7)

        # testing phase
        for j in range(len(imgs)):
            if (j >= test_range_down) and (j < test_range_up):
                img_test = imgs[j]
                ref_img = refs[j]
                a0 = np.multiply(w[0], img_test[:, :, 0]) + b[0]
                a1 = np.multiply(w[1], img_test[:, :, 1]) + b[1]
                a2 = np.multiply(w[2], img_test[:, :, 2]) + b[2]
                image_recovered = np.zeros(img_test.shape)
                image_recovered[:, :, 0] = a0
                image_recovered[:, :, 1] = a1
                image_recovered[:, :, 2] = a2
                ssim_val = ssim(ref_img, image_recovered, multichannel=True)
                psnr_val = psnr(ref_img, image_recovered)
                sum_ssim = sum_ssim + ssim_val
                sum_psnr = sum_psnr + psnr_val
                test_num += 1

        ssim_folds.append(sum_ssim)
        psnr_folds.append(sum_psnr)

    avg_psnr = sum(psnr_folds) / test_num
    avg_ssim = sum(ssim_folds) / test_num

    print('%d images are tested.\n' % test_num)

    return avg_psnr, avg_ssim


# DCP:
def dcp_method(imgs, refs, srcs):
    test_num = 0
    sum_ssim = 0
    sum_psnr = 0
    for j in range(len(imgs)):
        img = imgs[j]
        ref_img = refs[j]
        src = srcs[j]
        dark = dark_channel(img, PATCH_SIZE)
        a = atm_light(img, dark)
        t_estimated = transmission_estimate(img, a, PATCH_SIZE)
        t_refined = transmission_refine(src, t_estimated)
        recovered_img = recover(img, t_refined, a, 0.1)
        ssim_val = ssim(ref_img, recovered_img, multichannel=True)
        psnr_val = psnr(ref_img, recovered_img)
        sum_ssim = sum_ssim + ssim_val
        sum_psnr = sum_psnr + psnr_val
        test_num += 1

    avg_psnr = sum_psnr / test_num
    avg_ssim = sum_ssim / test_num

    print('%d images are tested.\n' % test_num)

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    data_dir = 'SOTS'
    in_or_out = 'outdoor'
    im_dir = os.path.join(data_dir, in_or_out, 'hazy')
    ref_dir = os.path.join(data_dir, in_or_out, 'gt')

    if not os.path.isdir(im_dir):
        print ('Given path %s not found' % format(im_dir))
        sys.exit(-1)

    if not os.path.isdir(ref_dir):
        print ('Given path %s not found' % format(ref_dir))
        sys.exit(-1)

    img_all = []
    ref_img_all = []
    src_all = []
    ref_src_all = []

    for fn in os.listdir(im_dir):
        im_path = os.path.join(im_dir, fn)
        fn_ref = fn[:4] + '.png'
        ref_path = os.path.join(ref_dir, fn_ref)
        assert (os.path.exists(im_path)), 'Annotation: %s does not exist' % format(im_path)
        assert (os.path.exists(ref_path)), 'Annotation: %s does not exist' % format(ref_path)
        ref_src = cv2.imread(ref_path)
        ref_img = ref_src.astype('float64') / 255
        src = cv2.imread(im_path)
        img = src.astype('float64') / 255
        img_all.append(img)
        ref_img_all.append(ref_img)
        src_all.append(src)
        ref_src_all.append(ref_src)

    print('%d images are loaded.' % len(img_all))

    if METHOD == 1:
        print('dehaze using learning method')
        psnr_mean, ssim_mean = train_method(img_all, ref_img_all)
    elif METHOD == 2:
        print('dehaze using DCP')
        psnr_mean, ssim_mean = dcp_method(img_all, ref_img_all, src_all)
    else:
        assert()

    print('The average PSNR value is %0.4f.\n' % psnr_mean)
    print('The average SSIM value is %0.4f.\n' % ssim_mean)

import os
import sys
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

# parameters are set here
PATCH_SIZE = 15
HAZE_WEIGHT = 0.6
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
IMG_START = 1
IMG_END = 3000
IMG_START_test = 1
IMG_END_test = 500


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


def propagate(w1, w2, w3, b, X, Y, t_refined, A, learning_rate):
    m, n = X.shape[0], X.shape[1]
    X1 = np.zeros(X.shape)
    X1[:,:,0] = np.divide(X[:,:,0], t_refined)
    X1[:,:,1] = np.divide(X[:,:,1], t_refined)
    X1[:,:,2] = np.divide(X[:,:,2], t_refined)
    X2 = np.zeros(X.shape)
    X2[:,:,0] = np.divide(A[0],t_refined)
    X2[:,:,1] = np.divide(A[1],t_refined)
    X2[:,:,2] = np.divide(A[2],t_refined)

    #Forward propagation
    Y0 = np.multiply(X1[:,:,0], w1[0]) + np.multiply(X2[:,:,0], w2[0]) + np.multiply(w3[0], A[0]) + b[0]
    Y1 = np.multiply(X1[:,:,1], w1[1]) + np.multiply(X2[:,:,1], w2[1]) + np.multiply(w3[1], A[1]) + b[1]
    Y2 = np.multiply(X1[:,:,2], w1[2]) + np.multiply(X2[:,:,2], w2[2]) + np.multiply(w3[2], A[2]) + b[2]

    #Backward propagation
    dw10 = np.sum(np.multiply(X1[:,:,0], (Y0 - Y[:,:,0])), dtype=np.float64) / (m*n)
    dw11 = np.sum(np.multiply(X1[:,:,1], (Y1 - Y[:,:,1])), dtype=np.float64) / (m*n)
    dw12 = np.sum(np.multiply(X1[:,:,2], (Y2 - Y[:,:,2])), dtype=np.float64) / (m*n)
    dw1 = [dw10, dw11, dw12]

    dw20 = np.sum(np.multiply(X2[:,:,0], (Y0 - Y[:,:,0])), dtype=np.float64) / (m*n)
    dw21 = np.sum(np.multiply(X2[:,:,1], (Y1 - Y[:,:,1])), dtype=np.float64) / (m*n)
    dw22 = np.sum(np.multiply(X2[:,:,2], (Y2 - Y[:,:,2])), dtype=np.float64) / (m*n)
    dw2 = [dw20, dw21, dw22]

    dw30 = np.sum(np.multiply(A[0], (Y0 - Y[:,:,0])), dtype=np.float64) / (m*n)
    dw31 = np.sum(np.multiply(A[1], (Y1 - Y[:,:,1])), dtype=np.float64) / (m*n)
    dw32 = np.sum(np.multiply(A[2], (Y2 - Y[:,:,2])), dtype=np.float64) / (m*n)
    dw3 = [dw30, dw31, dw32]

    db0 = np.sum(Y0 - Y[:,:,0], dtype = np.float64)/(m*n)
    db1 = np.sum(Y1 - Y[:,:,1], dtype = np.float64)/(m*n)
    db2 = np.sum(Y2 - Y[:,:,2], dtype = np.float64)/(m*n)
    db = [db0, db1, db2]

    w1 = w1 - np.multiply(learning_rate, dw1)
    w2 = w2 - np.multiply(learning_rate, dw2)
    w3 = w3 - np.multiply(learning_rate, dw3) 
    b = b - np.multiply(learning_rate, db)   

    return w1, w2, w3, b
    
if __name__ == '__main__':
    # if images file SOTS is not at current directory, please change the data_dir
    data_dir = 'images'
    in_or_out = 'outdoor'
    im_dir_train = os.path.join(data_dir, in_or_out, 'OTS', 'OTS')
    ref_dir_train = os.path.join(data_dir, in_or_out, 'clear_images')
    im_dir_test = os.path.join(data_dir, in_or_out, 'hazy')
    ref_dir_test = os.path.join(data_dir, in_or_out, 'gt')

    img_read = 0
    w1 = [0,0,0]
    w2 = [0,0,0]
    w3 = [0,0,0]
    b = [0,0,0]

    for fn in os.listdir(im_dir_train):
        img_read = img_read + 1
        if img_read < IMG_START:
            continue
        if img_read >= IMG_END:
            break
        im_path = os.path.join(im_dir_train, fn)
        fn_ref = fn[:4] +'.jpg'
        ref_path = os.path.join(ref_dir_train, fn_ref)
        assert(os.path.exists(im_path)), 'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(ref_path)),'Annotation: {} does not exist'.format(ref_path)
        ref_src = cv2.imread(ref_path)
        ref_img = ref_src.astype('float64')/255
        src = cv2.imread(im_path)
        img = src.astype('float64')/255

        dark = dark_channel(img, PATCH_SIZE)
        A = atm_light(img, dark)
        t_estimated = transmission_estimate(img, A, PATCH_SIZE)
        t_refined = transmission_refine(src, t_estimated)
        
        w1, w2, w3, b = propagate(w1, w2, w3, b, img, ref_img, t_refined, A,  learning_rate = 0.23)
    
    img_read_test = 0
    sum_ssim = 0
    sum_psnr = 0

    for x in os.listdir(im_dir_test):
        img_read_test = img_read_test + 1
        if img_read_test < IMG_START_test:
            continue 
        if img_read_test >= IMG_END_test:
            break
        im_path_test = os.path.join(im_dir_test, x)
        x_ref = x[:4] + '.png'
        ref_path_test = os.path.join(ref_dir_test, x_ref)
        assert(os.path.exists(im_path_test)), 'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(ref_path_test)), 'Annotation: {} does not exist'.format(ref_path)
        ref_src_test = cv2.imread(ref_path_test)
        ref_img_test = ref_src_test.astype('float64')/255
        src_test = cv2.imread(im_path_test)
        img_test = src_test.astype('float64')/255

        dark = dark_channel(img_test, PATCH_SIZE)
        A = atm_light(img_test, dark)
        t_estimated = transmission_estimate(img_test, A, PATCH_SIZE)
        t_refined = transmission_refine(src_test, t_estimated)

        Y0 = np.multiply(np.divide(img_test[:,:,0], t_refined), w1[0]) + np.multiply(np.divide(A[0],t_refined), w2[0]) + np.multiply(w3[0], A[0]) + b[0]
        Y1 = np.multiply(np.divide(img_test[:,:,1], t_refined), w1[1]) + np.multiply(np.divide(A[1],t_refined), w2[1]) + np.multiply(w3[1], A[1]) + b[1]
        Y2 = np.multiply(np.divide(img_test[:,:,2], t_refined), w1[2]) + np.multiply(np.divide(A[2],t_refined), w2[2]) + np.multiply(w3[2], A[2]) + b[2]
        image_recovered = np.zeros(img_test.shape)
        image_recovered[:,:,0] = Y0
        image_recovered[:,:,1] = Y1
        image_recovered[:,:,2] = Y2
        ssim_val = ssim(ref_img_test, image_recovered, multichannel=True)
        psnr_val = psnr(ref_img_test, image_recovered)
        sum_ssim = sum_ssim + ssim_val
        sum_psnr = sum_psnr + psnr_val

    avg_psnr = sum_psnr/(IMG_END_test - IMG_START_test)
    avg_ssim = sum_ssim/(IMG_END_test - IMG_START_test)


    print(w1, w2, w3, b)
    #print('The last SSIM value is %0.4f.\n', ssim_val)
    #print('The last PSNR value is %0.4f.\n', psnr_val)
    print('The average PSNR value is %0.4f.\n', avg_psnr)
    print('The average SSIM value is %0.4f.\n', avg_ssim)
    cv2.imshow('Original Image', img_test)
    cv2.imshow('Recovered Image', image_recovered)
    cv2.waitKey()

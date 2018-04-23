import os
import sys
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim

IMG_START = 1
IMG_END = 450
IMG_START_test = 450
IMG_END_test = 500

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def sigmoid_prime(z):
    s = sigmoid(z) * (1 - sigmoid(z))
    return s 

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

def propagate(w, b, X, Y, learning_rate):
    m, n  = X.shape[0], X.shape[1]
    
    #Forward propagation
    A0 = (np.multiply(w[0], X[:,:,0]) + b[0])
    A1 = (np.multiply(w[1], X[:,:,1]) + b[1])
    A2 = (np.multiply(w[2], X[:,:,2]) + b[2])
    #cost = (np.sum(np.multiply(Y[:,:,0],np.log(A0)) + np.multiply((np.ones((m,n)) - Y[:,:,0]), np.log(np.ones((m,n)) - A0)))+
    #      np.sum(np.multiply(Y[:,:,1],np.log(A1)) + np.multiply((np.ones((m,n)) - Y[:,:,1]), np.log(np.ones((m,n)) - A1)))+
    #       np.sum(np.multiply(Y[:,:,2],np.log(A2)) + np.multiply((np.ones((m,n)) - Y[:,:,2]), np.log(np.ones((m,n)) - A2))))*(-1/(m*n*3))
    
    #Backward propagation
    dw0 = np.sum(np.multiply((X[:,:,0]), (A0 - Y[:,:,0])), dtype = np.float64)/(m*n)
    dw1 = np.sum(np.multiply((X[:,:,1]), (A1 - Y[:,:,1])), dtype = np.float64)/(m*n)
    dw2 = np.sum(np.multiply((X[:,:,2]), (A2 - Y[:,:,2])), dtype = np.float64)/(m*n)
    #dw0 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[0], X[:,:,0])), (A0 - Y[:,:,0])), dtype = np.float64)/(m*n)
    #dw1 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[1], X[:,:,1])), (A0 - Y[:,:,1])), dtype = np.float64)/(m*n)
    #dw2 = np.sum(np.multiply(sigmoid_prime(np.multiply(w[2], X[:,:,2])), (A0 - Y[:,:,2])), dtype = np.float64)/(m*n)
    db0 = np.sum(A0 - Y[:,:,0], dtype = np.float64)/(m*n)
    db1 = np.sum(A1 - Y[:,:,1], dtype = np.float64)/(m*n)
    db2 = np.sum(A2 - Y[:,:,2], dtype = np.float64)/(m*n)
    
    #cost = np.squeeze(cost)
    dw = [dw0, dw1, dw2]
    db = [db0, db1, db2]
    w = w - np.multiply(learning_rate, dw)
    b = b - np.multiply(learning_rate, db)
    
    return w, b

if __name__ == '__main__':
    data_dir = 'images'
    in_or_out = 'outdoor'
    im_dir = os.path.join(data_dir, in_or_out, 'hazy')
    ref_dir = os.path.join(data_dir, in_or_out, 'gt')
    
    if not os.path.isdir(im_dir):
        print ('Given path {} not found', format(im_dir))
        sys.exit(-1)
    
    img_read = 0
    w = [0, 0, 0]
    b = [0, 0, 0]
    
    for fn in os.listdir(im_dir):
        img_read = img_read + 1
        if img_read < IMG_START:
            continue
        if img_read >= IMG_END:
            break
        im_path = os.path.join(im_dir, fn)
        fn_ref = fn[:4] +'.png'
        ref_path = os.path.join(ref_dir, fn_ref)
        assert(os.path.exists(im_path)), 'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(ref_path)),'Annotation: {} does not exist'.format(ref_path)
        ref_src = cv2.imread(ref_path)
        ref_img = ref_src.astype('float64')/255
        src = cv2.imread(im_path)
        img = src.astype('float64')/255

        w, b = propagate(w, b, img, ref_img, learning_rate = 0.7)

    img_read_test = 0
    sum_ssim = 0
    sum_psnr = 0

    for x in os.listdir(im_dir):
        img_read_test = img_read_test + 1
        if img_read_test < IMG_START_test:
            continue 
        if img_read_test >= IMG_END_test:
            break
        im_path_test = os.path.join(im_dir, x)
        x_ref = x[:4] + '.png'
        ref_path_test = os.path.join(ref_dir, x_ref)
        assert(os.path.exists(im_path_test)), 'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(ref_path_test)), 'Annotation: {} does not exist'.format(ref_path)
        ref_src_test = cv2.imread(ref_path_test)
        ref_img_test = ref_src_test.astype('float64')/255
        src_test = cv2.imread(im_path_test)
        img_test = src_test.astype('float64')/255

        A0 = np.multiply(w[0], img_test[:,:,0]) + b[0]
        A1 = np.multiply(w[1], img_test[:,:,1]) + b[1]
        A2 = np.multiply(w[2], img_test[:,:,2]) + b[2]
        image_recovered = np.zeros(img_test.shape)
        image_recovered[:,:,0] = A0
        image_recovered[:,:,1] = A1
        image_recovered[:,:,2] = A2
        ssim_val = ssim(image_recovered, ref_img_test, multichannel=True)
        psnr_val = psnr(image_recovered, ref_img_test)
        sum_ssim = sum_ssim + ssim_val
        sum_psnr = sum_psnr + psnr_val

    avg_psnr = sum_psnr/(IMG_END_test - IMG_START_test)
    avg_ssim = sum_ssim/(IMG_END_test - IMG_START_test)

    #print (image_recovered)
    print('%d images are tested.\n', IMG_END_test - IMG_START_test)
    #print('The last SSIM value is %0.4f.\n', ssim_val)
    #print('The last PSNR value is %0.4f.\n', psnr_val)
    print('The average PSNR value is %0.4f.\n', avg_psnr)
    print('The average SSIM value is %0.4f.\n', avg_ssim)
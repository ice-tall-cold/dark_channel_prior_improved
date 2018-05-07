# Improved dehazed algorithm based on Dark Channel Prior

## introduction

### Route setting
directory for training haze images: OTS/haze  
directory for training groundtruth images: OTS/gt
directory for testing outdoor haze images: SOTS/outdoor/haze  
directory for testing outdoor groundtruth images: SOTS/outdoor/gt
directory for testing indoor haze images: SOTS/indoor/haze  
directory for testing indoor groundtruth images: SOTS/indoor/gt  

***Please move all the images to required directories before running the code.***

### Define setting
In the Main.py, functions are defined at the beginning:  

```
DETECT_SKY = 0
RECOVER_WHITENING = 0
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 0
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 2  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test
```
DETECT_SKY enable the function to seperate image into sky region and non-sky region.  

RECOVER_WHITENING enable the tone mapping method.  

BRIGHTEST_IN_DC is the most common way to estimate atmospheric light, so always set it to 1.  

GUIDED_FILTER enable the guided filter function.  

INVERSE_SKY enable the function to estimate t(x) of sky region on inverse image.  

ENABLE_PRINT_FINAL_IMAGE enable printing out results as well as saving recovered image.

METHOD 1 is DCP method to recover image; METHOD 2 is training method to recover image; METHOD 3 simply shows the sky segmentation results.
  
## Using method
For original DCP method, the setting is:  

```
DETECT_SKY = 0
RECOVER_WHITENING = 0
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 0
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 1  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test
```  

For Sky segmentation method, the setting is:  

```
DETECT_SKY = 1
RECOVER_WHITENING = 1
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 1
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 1  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test
```

For Training method, the setting is:  

```
DETECT_SKY = 1
RECOVER_WHITENING = 1
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 1
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 2  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test
```

Training method can also tested individually, the setting is:  

```
DETECT_SKY = 0
RECOVER_WHITENING = 0
BRIGHTEST_IN_DC = 1
GUIDED_FILTER = 1
INVERSE_SKY = 0
ENABLE_PRINT_FINAL_IMAGE = 1

METHOD = 2  # 1: DCP; 2: Train_Method; 3: Simple SKY_SEGMENTATION test
```

## Information
Authors: Wenrui Zhang & Binghan Li  (TAMU CSCE 633)
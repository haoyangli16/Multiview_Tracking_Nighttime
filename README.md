# Multi-View Domain Adaptation for Nighttime Aerial Tracking. 

[Haoyang Li](https://vision4robotics.github.io/), [Changhong Fu](https://vision4robotics.github.io/authors/changhong-fu/), [Guangze Zheng](https://zhengguangze.netlify.app/), [Sihang Li](https://vision4robotics.github.io/), and [Junjie Ye](https://jayye99.github.io/). Multi-View Domain Adaptation for Nighttime Aerial Tracking. 


![overview](https://github.com/LiHaoyang0616/MVDANT/blob/main/fig/overview.png)

## Overview

**MVDANT** is an unsupervised domain adaptation framework for visual object tracking. This repo contains its Python implementation.

## Testing MVDANT

### 1. Preprocessing

Before training, we need to preprocess the unlabelled training data to generate training pairs.

1. Download the proposed [NAT2021-*train* set](https://vision4robotics.github.io/NAT2021/)

2. Customize the directory of the train set in `lowlight_enhancement.py` and enhance the nighttime sequences

   ```python
   cd preprocessing/
   python lowlight_enhancement.py # enhanced sequences will be saved at '/YOUR/PATH/NAT2021/train/data_seq_enhanced/'
   ```

3. Download the video saliency detection model [here](https://drive.google.com/file/d/1Fuw3oC86AqZhH5F3pko_aqAMhPtQyt6j/view?usp=sharing) and place it at `preprocessing/models/checkpoints/`.

4. Predict salient objects and obtain candidate boxes

   ``` python
   python inference.py # candidate boxes will be saved at 'coarse_boxes/' as .npy
   ```

5. Generate pseudo annotations from candidate boxes using dynamic programming

   ``` python
   python gen_seq_bboxes.py # pseudo box sequences will be saved at 'pseudo_anno/'
   ```

6. Generate cropped training patches and a JSON file for training

   ``` py
   python par_crop.py
   python gen_json.py
   ```

### 2. Train

Take MVDANT for instance.

1. Apart from above target domain dataset NAT2021, you need to download and prepare source domain datasets [VID](https://image-net.org/challenges/LSVRC/2017/) and [GOT-10K](http://got-10k.aitestunion.com/downloads).

2. Download the pre-trained daytime model ([SiamCAR](https://drive.google.com/drive/folders/11Jimzxj9QONOACJBKzMQ9La6GZhA73QD?usp=sharing)/[SiamBAN](https://drive.google.com/drive/folders/17Uz3dZFOtx-uU7J4t48_nAfPXvNsQAAq?usp=sharing)) and place it at `UDAT/tools/snapshot`.

3. Start training

   ``` python
   cd MVADNT
   export PYTHONPATH=$PWD
   python tools/train_MVADNT.py
   ```

### 3. Test
Take MVADNT for instance.
1. For quick test, you can download our trained model for [MVADNT](https://drive.google.com/file/d/1DccbQ4nh2rlni8RVykTNzuHXJgSvNE4G/view?usp=sharing) and place it at `MVADNT/CAR/experiments/udatcar_r50_l234`.

2. Start testing

    ```python
    python tools/test.py --dataset NAT 
    python tools/test.py --dataset UAVDark70
    ```

### 4. Eval

1. Start evaluating
    ``` python
    python tools/eval.py --dataset NAT
    python tools/eval.py --dataset UAVDark70
    ```

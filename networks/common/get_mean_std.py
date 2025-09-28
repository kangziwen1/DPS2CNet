import os
import random

import cv2
import numpy as np

# calculate means and std
train_txt_path = './data/Label/TR.txt'
base_path = './data/food'

CNum = 66071  # 挑选多少图片进行计算

img_h, img_w = 256, 256
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)  # shuffle , 随机挑选图片

    for i in range(CNum):
        # img_path = os.path.join(base_path, lines[i].rstrip().split()[0])
        img_path = base_path + lines[i].rstrip().split()[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


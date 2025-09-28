# -*- coding: utf-8 -*-

# @Time : 2022-11-11

# @Author : gyf

# @File : Get_semantic_kitti_mini.py

# -*- 功能说明 -*-

# 采样SemanticKITTI的1/10

# -*- 功能说明 -*-
import random
import os
import shutil
from glob import glob

from tqdm import tqdm


def print_info(info: str):
    print('\n')
    print("\033[91m========================={}=========================\033[0m".format(info))
    print('\n')


def get_semantic_mini(orgionPath, targetPath):
    sequences = os.listdir(orgionPath)
    content = ['velodyne', 'labels', 'voxels', 'image_2']
    extension = ['.bin', '.label', '.invalid', '.occluded', '.png']
    sequences = sequences[:11]
    for sequence in sequences:  # 0-10 切一下    val -> 1/7  other1/15
        filesNames = os.listdir(os.path.join(orgionPath, sequence, content[0]))
        # glob(os.path.join(orgionPath, sequence, content[0]),'*.bin')
        for index in range(len(filesNames)):
            filesNames[index] = filesNames[index].split('.')[0]

        # region 1/10 采样
        random.seed(3407)
        random.shuffle(filesNames)
        random.seed(3407)
        if sequence == '08':
            fileNamesSample = random.sample(filesNames, len(filesNames) // 7)
        else:
            fileNamesSample = random.sample(filesNames, len(filesNames) // 7)
        # endregion

        # ====================process velodyne====================

        print_info("processing velodyne at sequence: {}".format(sequence))
        orgionVelodyne = os.path.join(orgionPath, sequence, content[0])
        targetVelodyne = os.path.join(targetPath, sequence, content[0])
        if not os.path.exists(targetVelodyne):
            os.makedirs(targetVelodyne)
        for filename in tqdm(fileNamesSample):
            shutil.copyfile(os.path.join(orgionVelodyne, filename + extension[0])
                            , os.path.join(targetVelodyne, filename + extension[0]))

        # ====================process labels====================
        print_info("processing labels at sequence: {}".format(sequence))
        orgionLabels = os.path.join(orgionPath, sequence, content[1])
        targetLabels = os.path.join(targetPath, sequence, content[1])
        if not os.path.exists(targetLabels):
            os.makedirs(targetLabels)
        for filename in tqdm(fileNamesSample):
            shutil.copyfile(os.path.join(orgionLabels, filename + extension[1])
                            , os.path.join(targetLabels, filename + extension[1]))

        # ====================process voxels====================
        print_info("processing voxels at sequence: {}".format(sequence))
        orgionVoxels = os.path.join(orgionPath, sequence, content[2])
        targetvoxels = os.path.join(targetPath, sequence, content[2])
        if not os.path.exists(targetvoxels):
            os.makedirs(targetvoxels)
        for filename in tqdm(fileNamesSample):
            shutil.copyfile(os.path.join(orgionVoxels, filename + extension[0])
                            , os.path.join(targetvoxels, filename + extension[0]))
            shutil.copyfile(os.path.join(orgionVoxels, filename + extension[1])
                            , os.path.join(targetvoxels, filename + extension[1]))
            shutil.copyfile(os.path.join(orgionVoxels, filename + extension[2])
                            , os.path.join(targetvoxels, filename + extension[2]))
            shutil.copyfile(os.path.join(orgionVoxels, filename + extension[3])
                            , os.path.join(targetvoxels, filename + extension[3]))

        # ====================process images====================
        print_info("processing image_2 at sequence: {}".format(sequence))
        orgionImages = os.path.join(orgionPath, sequence, content[3])
        targetImages = os.path.join(targetPath, sequence, content[3])
        if not os.path.exists(targetImages):
            os.makedirs(targetImages)
        for filename in tqdm(fileNamesSample):
            shutil.copyfile(os.path.join(orgionImages, filename + extension[4])
                            , os.path.join(targetImages, filename + extension[4]))
    print("\033[92m=========================OVER ! ! !=========================\033[0m")

def get_semantic_rare(orgionPath, targetPath,filename,sequence):
    sequences = os.listdir(orgionPath)
    content = ['velodyne', 'labels', 'voxels', 'image_2']
    extension = ['.bin', '.label', '.invalid', '.occluded', '.png']

    # ====================process velodyne====================

    # print_info("processing velodyne at sequence: {}".format(sequence))
    orgionVelodyne = os.path.join(orgionPath, sequence, content[0])
    targetVelodyne = os.path.join(targetPath, sequence, content[0])
    if not os.path.exists(targetVelodyne):
        os.makedirs(targetVelodyne)

    shutil.copyfile(os.path.join(orgionVelodyne, filename + extension[0])
                    , os.path.join(targetVelodyne, filename + extension[0]))

    # ====================process labels====================
    # print_info("processing labels at sequence: {}".format(sequence))
    orgionLabels = os.path.join(orgionPath, sequence, content[1])
    targetLabels = os.path.join(targetPath, sequence, content[1])
    if not os.path.exists(targetLabels):
        os.makedirs(targetLabels)

    shutil.copyfile(os.path.join(orgionLabels, filename + extension[1])
                    , os.path.join(targetLabels, filename + extension[1]))

    # ====================process voxels====================
    # print_info("processing voxels at sequence: {}".format(sequence))
    orgionVoxels = os.path.join(orgionPath, sequence, content[2])
    targetvoxels = os.path.join(targetPath, sequence, content[2])
    if not os.path.exists(targetvoxels):
        os.makedirs(targetvoxels)

    shutil.copyfile(os.path.join(orgionVoxels, filename + extension[0])
                    , os.path.join(targetvoxels, filename + extension[0]))
    shutil.copyfile(os.path.join(orgionVoxels, filename + extension[1])
                    , os.path.join(targetvoxels, filename + extension[1]))
    shutil.copyfile(os.path.join(orgionVoxels, filename + extension[2])
                    , os.path.join(targetvoxels, filename + extension[2]))
    shutil.copyfile(os.path.join(orgionVoxels, filename + extension[3])
                    , os.path.join(targetvoxels, filename + extension[3]))

    # ====================process images====================
    # print_info("processing image_2 at sequence: {}".format(sequence))
    orgionImages = os.path.join(orgionPath, sequence, content[3])
    targetImages = os.path.join(targetPath, sequence, content[3])
    if not os.path.exists(targetImages):
        os.makedirs(targetImages)

    shutil.copyfile(os.path.join(orgionImages, filename + extension[4])
                    , os.path.join(targetImages, filename + extension[4]))
    # print("\033[92m=========================OVER ! ! !=========================\033[0m")


if __name__ == '__main__':
    get_semantic_mini \
            (
            orgionPath='/media/qq/0CBE052B0CBE052B/dataset/GYF/SemanticKITTI/data_odometry_velodyne/dataset/sequences/',
            targetPath='/media/qq/0CBE052B0CBE052B/dataset/SemanticKTTTIMini/dataset/sequences'
        )

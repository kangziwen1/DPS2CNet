import os

import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from get_semantic_kitti_mini import get_semantic_rare
from networks.common import metrics
from networks.data.io_data import _read_SemKITTI, _read_invalid_SemKITTI

yaml_path = '/home/qq/code/gyf/SSA-SC-MREF/networks/data'
dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'semantic-kitti.yaml'), 'r'))


def _read_label_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False).astype(np.float32)
    return label


def get_remap_lut():
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut


def read_label(path, invalid_path):
    LABEL = _read_label_SemKITTI(path)
    LABEL = get_remap_lut()[LABEL.astype(np.uint16)].astype(np.float32)
    INVALID = _read_invalid_SemKITTI(invalid_path)
    LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
    # LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
    #                                    int(self.grid_dimensions[2] / scale_divide),
    #                                    int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])
    return LABEL


def print_info(info: str):
    print('\n')
    print("\033[91m========================={}=========================\033[0m".format(info))
    print('\n')


def get(JustVal=True):
    dict = {}
    for i in range(1, 20):
        class_name = dataset_config['labels'][dataset_config['learning_map_inv'][i]]
        dict[class_name] = i
    pass

    numList = [0 for i in range(1, 20)]
    # numList = np.zeros(19)

    dataset_path = '/media/qq/0CBE052B0CBE052B/dataset/SemanticKTTTIMini/dataset/sequences'
    sequences = os.listdir(dataset_path)
    content = ['voxels']
    extension = ['.label', '.invalid']
    sequences = sequences[:11]
    for sequence in sequences:  # 0-10 切一下
        if JustVal:
            if sequence != '08':
                continue
        elif sequence == '08':
            continue
        current_folder = os.path.join(dataset_path, sequence, content[0])
        filesNames = os.listdir(current_folder)
        for index in range(len(filesNames)):
            filesNames[index] = filesNames[index].split('.')[0]
        print_info('Processing sequence: {} !!!!'.format(sequence))
        for filesName in tqdm(filesNames):
            label = read_label(os.path.join(current_folder, filesName + extension[0]),
                               os.path.join(current_folder, filesName + extension[1])).astype(int)
            for i in range(len(numList)):
                index = i+1
                numList[i] += np.sum(label[label == index])
        # percentage = numList / sum(numList)
    plt.barh(list(dict.keys()), numList)
    plt.show()
    if JustVal:
        plt.savefig('./stat-val.jpg')
    else:
        plt.savefig('./stat-train.jpg')

    sumNum = sum(numList)
    percentage = [x/sumNum for x in numList]
    percentageList = []
    for p in percentage:
        p = str(round(p*100,2)) + '%'
        percentageList.append(p)
    print(percentageList)

def get_rare():
    dict = {}
    for i in range(1, 20):
        class_name = dataset_config['labels'][dataset_config['learning_map_inv'][i]]
        dict[class_name] = i
    pass

    numList = [0 for i in range(1, 20)]
    # numList = np.zeros(19)

    dataset_path = '/media/qq/0CBE052B0CBE052B/dataset/GYF/SemanticKITTI/data_odometry_velodyne/dataset/sequences'
    # dataset_path = '/media/qq/0CBE052B0CBE052B/dataset/SemanticKTTTIMini/dataset/sequences'

    target_path = '/media/qq/0CBE052B0CBE052B/dataset/SemanticKITTIRare/dataset/sequences'
    sequences = os.listdir(dataset_path)
    content = ['voxels']
    extension = ['.label', '.invalid']
    sequences = sequences[:11]
    # rare_class_name = ['bicycle','motorcycle','truck''other-vehicle','person','other-ground','bicyclist','motorcyclist','traffic-sign']
    rare_class_name = ['motorcyclist']

    rare_class_index = []
    for key in dict.keys():
        if key in rare_class_name:
            rare_class_index.append(dict[key])

    for sequence in sequences:  # 0-10 切一下
        current_folder = os.path.join(dataset_path, sequence, content[0])
        filesNames = os.listdir(current_folder)
        for index in range(len(filesNames)):
            filesNames[index] = filesNames[index].split('.')[0]
        print_info('Processing sequence: {} !!!!'.format(sequence))
        for filesName in tqdm(filesNames):
            label = read_label(os.path.join(current_folder, filesName + extension[0]),
                               os.path.join(current_folder, filesName + extension[1])).astype(int)
            for index in rare_class_index:
                if np.sum(label[label == index]) != 0:
                    get_semantic_rare(dataset_path,target_path,filesName,sequence)
                    continue



if __name__ == '__main__':
     # get(True)
    # dict = {}
    # per = ['0.31%', '0.01%', '0.01%', '0.02%', '0.1%', '0.13%', '0.22%', '0.0%', '7.96%', '0.68%', '6.46%', '0.06%', '13.36%', '1.26%', '48.33%', '1.08%', '19.51%', '0.39%', '0.11%']
    # for i in range(1,20):
    #     class_name = dataset_config['labels'][dataset_config['learning_map_inv'][i]]
    #     dict[class_name] = per
    # index = 0
    # for key in dict.keys():
    #     dict[key] = per[index]
    #     index +=1
    #
    # print(dict)
    get_rare()

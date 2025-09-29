from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import yaml
import random
import sys
import numba as nb
import torch
from torchvision import transforms
import networks.data.io_data as SemanticKittiIO


def collate_fn_BEV(data):
    needRGB = False
    RGB = None
    OCCUPANCY = np.stack([d[0]['3D_OCCUPANCY'] for d in data])
    LABEL = np.stack([d[0]['3D_LABEL']['1_1'] for d in data])
    feature = [d[0]['3D_POINT']['feature'] for d in data]
    voxel_ind = [d[0]['3D_POINT']['voxel_ind'] for d in data]
    # processed_label = np.stack([d[0]['3D_POINT']['processed_label'] for d in data])
    if needRGB:
        RGB = torch.stack([d[0]['2D_RGB'] for d in data])  # change
    idx = [d[1] for d in data]

    # data = {'3D_OCCUPANCY': torch.from_numpy(OCCUPANCY), 'feature': feature,
    #         'voxel_ind': voxel_ind, 'processed_label': torch.from_numpy(processed_label), '2D_RGB': RGB}

    data = {'3D_OCCUPANCY': torch.from_numpy(OCCUPANCY), 'feature': feature,
            'voxel_ind': voxel_ind,  '2D_RGB': RGB}

    data['3D_LABEL'] = {'1_1': torch.from_numpy(LABEL)}

    return data, idx


def collate_fn_BEV_test(data):
    OCCUPANCY = np.stack([d[0]['3D_OCCUPANCY'] for d in data])
    OCCLUDED = np.array([])
    LABEL = np.array([])
    feature = [d[0]['3D_POINT']['feature'] for d in data]
    voxel_ind = [d[0]['3D_POINT']['voxel_ind'] for d in data]
    processed_label = np.array([])
    idx = [d[1] for d in data]

    data = {'3D_OCCUPANCY': torch.from_numpy(OCCUPANCY),  'feature': feature,
            'voxel_ind': voxel_ind, 'processed_label': torch.from_numpy(processed_label)}
    data['3D_LABEL'] = {'1_1': torch.from_numpy(LABEL)}

    return data, idx


class Voxelizer:
    def __init__(self, grid_size, ignore_label=255, return_test=True, fixed_volume_space=False,
                 max_volume_space=[51.2, 25.6, 4.4], min_volume_space=[0, -25.6, -2]):
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.trans_std = [0.1, 0.1, 0.1]

    def voxelization(self, data, flip_type=0, phase='train'):
        'Generates one sample of data'

        # kzw
        # xyz, labels, sig = data
        xyz, sig = data
        if len(sig.shape) == 2:
            sig = np.squeeze(sig)



        # random data augmentation by flip x , y or x+y
        if flip_type:
            if np.isclose(flip_type, 1):
                xyz[:, 0] = 51.2 - xyz[:, 0]
            elif np.isclose(flip_type, 2):
                xyz[:, 1] = -xyz[:, 1]
            elif np.isclose(flip_type, 3):
                xyz[:, 0] = 51.2 - xyz[:, 0]
                xyz[:, 1] = -xyz[:, 1]


        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        for i in range(3):
            index = np.where((xyz[:, i] > min_bound[i]) & (xyz[:, i] < max_bound[i]))
            xyz = xyz[index]
            sig = sig[index]
            # labels = labels[index]

        voxel_ind = (np.floor((xyz - min_bound) / intervals)).astype(np.int)

        # process labels
        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        # label_voxel_pair = np.concatenate([voxel_ind, labels], axis=1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((voxel_ind[:, 0], voxel_ind[:, 1], voxel_ind[:, 2])), :]
        # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_dict = {}

        # center data on each voxel for PTnet
        voxel_centers = (voxel_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)  # 3+3+1
        data_dict['feature'] = return_fea
        data_dict['voxel_ind'] = voxel_ind
        # data_dict['processed_label'] = processed_label

        return data_dict

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

class CyliderVoxelizer:
    def __init__(self, grid_size, ignore_label=255, return_test=True, fixed_volume_space=False,
                 max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4], ):
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.trans_std = [0.1, 0.1, 0.1]

    def voxelization(self, data, flip_type=0, phase='train'):
        xyz, labels, sig = data
        if len(sig.shape) == 2:
            sig = np.squeeze(sig)

        if phase == 'train':  # Rotation
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if flip_type:
            if np.isclose(flip_type, 1):
                xyz[:, 0] = 51.2 - xyz[:, 0]
            elif np.isclose(flip_type, 2):
                xyz[:, 1] = -xyz[:, 1]
            elif np.isclose(flip_type, 3):
                xyz[:, 0] = 51.2 - xyz[:, 0]
                xyz[:, 1] = -xyz[:, 1]

        if phase == 'train':  # scale_aug
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if phase == 'train':  # noise_transform
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # data_tuple = (voxel_position, processed_label)

        data_dict = {}

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
        return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)


        data_dict['feature'] = return_fea
        data_dict['voxel_ind'] = grid_ind
        data_dict['processed_label'] = processed_label

        return data_dict


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


class SemanticKITTI_dataloader(Dataset):
    def __init__(self, dataset, phase):
        '''

        :param dataset: The dataset configuration (data augmentation, input encoding, etc)
        :param phase_tag: To differentiate between training, validation and test phase
        '''

        yaml_path, _ = os.path.split(os.path.realpath(__file__))
        self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'ssc-kitti360.yaml'), 'r'))
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims']  # [W, H, D]
        self.remap_lut = self.get_remap_lut()
        self.rgb_mean = np.array(
            [0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
        self.rgb_std = np.array(
            [0.30599035, 0.3129534, 0.31933814])  # images std:  [78.02753826 79.80311686 81.43122464]
        self.root_dir = dataset['ROOT_DIR']
        self.modalities = dataset['MODALITIES']
        self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded',
                           '3D_INVALID': '.invalid', '3D_POINT': '.bin', '2D_RGB': '.png'}
        self.data_augmentation = {'FLIPS': dataset['AUGMENTATION']['FLIPS']}

        self.filepaths = {}
        self.phase = phase
        self.class_frequencies = np.array([
            5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05,
            2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06,
            4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06,
            3.69705220e+07, 1.15198800e+06, 3.34146000e+05
        ])




        self.split = {'train': [0,1,2,3,4,5,6],
                          'val': [8],
                      'test': [8]}  # change 'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        for modality in self.modalities:
            if self.modalities[modality]:
                self.get_filepaths(modality)

        self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])
        self.voxelizer = Voxelizer(grid_size=[256, 256, 32], ignore_label=255, return_test=False,
                                   fixed_volume_space=True, max_volume_space=[51.2, 25.6, 4.4],
                                   min_volume_space=[0, -25.6, -2])
        # self.voxelizer = CyliderVoxelizer(grid_size=[256, 256, 32], ignore_label=255,return_test=False,
        #                            fixed_volume_space=True)

        return

    def get_filepaths(self, modality):
        '''
        Set modality filepaths with split according to phase (train, val, test)
        '''

        sequences = list(
            sorted(glob(os.path.join(self.root_dir, 'dataset', 'sequences', '*')))[i] for i in self.split[self.phase])

        if self.phase != 'test':

            if modality == '3D_LABEL':
                self.filepaths['3D_LABEL'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
                self.filepaths['3D_INVALID'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
                for sequence in sequences:
                    assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
                    # Scale 1:1
                    self.filepaths['3D_LABEL']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label')))
                    self.filepaths['3D_INVALID']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid')))
                    # Scale 1:2
                    self.filepaths['3D_LABEL']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_2')))
                    self.filepaths['3D_INVALID']['1_2'] += sorted(
                        glob(os.path.join(sequence, 'voxels', '*.invalid_1_2')))
                    # Scale 1:4
                    self.filepaths['3D_LABEL']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_4')))
                    self.filepaths['3D_INVALID']['1_4'] += sorted(
                        glob(os.path.join(sequence, 'voxels', '*.invalid_1_4')))
                    # Scale 1:8
                    self.filepaths['3D_LABEL']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_8')))
                    self.filepaths['3D_INVALID']['1_8'] += sorted(
                        glob(os.path.join(sequence, 'voxels', '*.invalid_1_8')))



        if modality == '3D_OCCUPANCY':
            self.filepaths['3D_OCCUPANCY'] = []
            for sequence in sequences:
                assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
                self.filepaths['3D_OCCUPANCY'] += sorted(glob(os.path.join(sequence, 'voxels', '*.bin')))

        if modality == '3D_POINT':
            self.filepaths['3D_POINT'] = []

        if modality == '2D_RGB':
            self.filepaths['2D_RGB'] = []
            for sequence in sequences:
                assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
                self.filepaths['2D_RGB'] += sorted(glob(os.path.join(sequence, 'image_2', '*.png')))

        return

    def check_same_nbr_files(self):
        '''
        Set modality filepaths with split according to phase (train, val, test)
        '''

        # TODO: Modify for nested dictionaries...
        for i in range(len(self.filepaths.keys()) - 1):
            length1 = len(self.filepaths[list(self.filepaths.keys())[i]])
            length2 = len(self.filepaths[list(self.filepaths.keys())[i + 1]])
            assert length1 == length2, 'Error: {} and {} not same number of files'.format(
                list(self.filepaths.keys())[i], list(self.filepaths.keys())[i + 1])
        return

    def __getitem__(self, idx):

        data = {}

        do_flip = 0
        if self.data_augmentation['FLIPS'] and self.phase == 'train':
            do_flip = random.randint(0, 3)

        for modality in self.modalities:
            if (self.modalities[modality]) and (modality in self.filepaths):
                data[modality] = self.get_data_modality(modality, idx, do_flip)

        return data, idx

    def get_data_modality(self, modality, idx, flip):

        if modality == '3D_OCCUPANCY':
            OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(self.filepaths[modality][idx])
            OCCUPANCY = np.moveaxis(
                OCCUPANCY.reshape([self.grid_dimensions[0], self.grid_dimensions[2], self.grid_dimensions[1]]),
                [0, 1, 2], [0, 2, 1])
            OCCUPANCY = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCUPANCY)
            return OCCUPANCY[None, :, :, :]

        elif modality == '3D_LABEL':
            LABEL_1_1 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_1', idx))
            return {'1_1': LABEL_1_1}


        elif modality == '3D_POINT':
            path = self.filepaths['3D_OCCUPANCY'][idx].replace('voxels', 'velodyne')
            POINT = SemanticKittiIO._read_point_SemKITTI(path)  # (N, 4)
            # path = self.filepaths['3D_OCCUPANCY'][idx].replace('voxels', 'labels').replace('.bin', '.label')
            # if self.phase != 'test':
            #     POINT_LABEL = SemanticKittiIO._read_label2_SemKITTI(path)
            #     POINT_LABEL = np.vectorize(self.dataset_config['learning_map'].__getitem__)(POINT_LABEL).astype(
            #         np.uint8)
            #     POINT_LABEL = POINT_LABEL - 1
            # else:
            #     POINT_LABEL = np.expand_dims(np.zeros_like(POINT[:, 0], dtype=int), axis=1)
            # VOXEL = self.voxelizer.voxelization([POINT[:, 0:3], POINT_LABEL, POINT[:, 3]], flip, self.phase)

            VOXEL = self.voxelizer.voxelization([POINT[:, 0:3], POINT[:, 3]], flip, self.phase)

            return VOXEL

        elif modality == '2D_RGB':
            # RGB = SemanticKittiIO._read_rgb_SemKITTI(self.filepaths[modality][idx])
            img = Image.open(self.filepaths[modality][idx])
            # mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
            if self.phase == 'train':
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.35),
                    transforms.RandomVerticalFlip(p=0.35),
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            # # TODO Standarize, Normalize RGB = SemanticKittiIO.img_normalize(RGB, self.rgb_mean, self.rgb_std) RGB
            #  = np.moveaxis(RGB, (0, 1, 2), (1, 2, 0)).astype(dtype='float32')  # reshaping [3xHxW] There is a
            #  problem on the RGB images.. They are not all the same size and I used those to calculate the mapping
            #  for the sketch... I need images all te same size..
            RGB = transform(img)
            return RGB

        else:
            assert False, 'Specified modality not found'

    def get_label_at_scale(self, scale, idx):

        scale_divide = int(scale[-1])
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][scale][idx])
        LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][scale][idx])
        if scale == '1_1':
            LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
        LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
        LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                           int(self.grid_dimensions[2] / scale_divide),
                                           int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

        return LABEL

    def read_semantics_config(self, data_path):

        # get number of interest classes, and the label mappings
        DATA = yaml.safe_load(open(data_path, 'r'))
        self.class_strings = DATA["labels"]
        self.class_remap = DATA["learning_map"]
        self.class_inv_remap = DATA["learning_map_inv"]
        self.class_ignore = DATA["learning_ignore"]
        self.n_classes = len(self.class_inv_remap)

        return

    def get_inv_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map_inv'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(
            self.dataset_config['learning_map_inv'].values())

        return remap_lut

    def get_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def __len__(self):
        """
        Returns the length of the dataset
        """
        # Return the number of elements in the dataset
        return self.nbr_files

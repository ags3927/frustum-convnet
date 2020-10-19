''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import pickle
import sys
import os
import numpy as np

import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs.config import cfg

from datasets.data_utils import rotate_pc_along_y, project_image_to_rect, compute_box_3d, extract_pc_in_box3d, roty
# from datasets.dataset_info import KITTICategory
from datasets.dataset_info import DATASET_INFO

logger = logging.getLogger(__name__)


class ProviderDataset(Dataset):

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False,
                 one_hot=True,
                 from_rgb_detection=False,
                 overwritten_data_path='',
                 extend_from_det=False):

        super(ProviderDataset, self).__init__()
        self.npoints = npoints
        self.split = split
        self.random_flip = random_flip
        self.random_shift = random_shift

        self.one_hot = one_hot
        self.from_rgb_detection = from_rgb_detection
        # MODIFICATION START - error distribution of heuristic
        self.errorMargins = np.empty(0)
        self.printPercentiles = False
        # MODIFICATION END - error distribution of heuristic

        dataset_name = cfg.DATA.DATASET_NAME
        assert dataset_name in DATASET_INFO
        self.category_info = DATASET_INFO[dataset_name]

        root_data = cfg.DATA.DATA_ROOT
        car_only = cfg.DATA.CAR_ONLY
        people_only = cfg.DATA.PEOPLE_ONLY

        if not overwritten_data_path:
            if not from_rgb_detection:
                if car_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_caronly_%s.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(root_data, 'frustum_carpedcyc_%s.pickle' % (split))
            else:
                if car_only:
                    overwritten_data_path = os.path.join(root_data,
                                                         'frustum_caronly_%s_rgb_detection.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s_rgb_detection.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(
                        root_data, 'frustum_carpedcyc_%s_rgb_detection.pickle' % (split))

        if from_rgb_detection:

            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)

        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.gt_box2d_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)
                # MODIFICATION START - load ground truth centers
                self.center_3d_raw_kitti = pickle.load(fp)
                # MODIFICATION END - load ground truth centers

            if extend_from_det:
                extend_det_file = overwritten_data_path.replace('.', '_det.')
                assert os.path.exists(extend_det_file), extend_det_file
                with open(extend_det_file, 'rb') as fp:
                    # extend
                    self.id_list.extend(pickle.load(fp))
                    self.box2d_list.extend(pickle.load(fp))
                    self.box3d_list.extend(pickle.load(fp))
                    self.input_list.extend(pickle.load(fp))
                    self.label_list.extend(pickle.load(fp))
                    self.type_list.extend(pickle.load(fp))
                    self.heading_list.extend(pickle.load(fp))
                    self.size_list.extend(pickle.load(fp))
                    self.frustum_angle_list.extend(pickle.load(fp))
                    self.gt_box2d_list.extend(pickle.load(fp))
                    self.calib_list.extend(pickle.load(fp))
                    # MODIFICATION START - load ground truth centers
                    self.center_3d_raw_kitti.extend(pickle.load(fp))
                    # MODIFICATION END - load ground truth centers
                logger.info('load dataset from {}'.format(extend_det_file))

        logger.info('load dataset from {}'.format(overwritten_data_path))

    def __len__(self):
        return len(self.input_list)

    # MODIFICATION START - HEURISTIC FUNCTIONS

    def regress_to_find_center(self, point_cloud_arr, stride = .5):
        # print(point_cloud_arr.shape)
        # sorted_arr = point_cloud_arr[np.argsort(point_cloud_arr[:,2])]

        freq_dict = defaultdict(int)
        freq_dict_aggregate = defaultdict(int)
        
        for idx in range(len(point_cloud_arr)):
            bin_number = point_cloud_arr[idx][2] // stride
            freq_dict[bin_number] += 1 

        max_key = max(freq_dict, key=freq_dict.get)
    
        # print(float(max_key) * stride)
        return max_key * stride

    def regress_to_find_center_aggregate(self, point_cloud_arr, stride = 0.5):
        freq_dict = defaultdict(int)
        # key = discrete z distance steps.
        # value = Number of pcl points that has z values at that step or bin.
        # bins are basically discrete z distance steps. Each step is separated by a stride of 0.5.
        # point_cloud_arr[idx][2] // stride finds exactly which "step" that certain point cloud goes to.
        # We also add that point cloud value to the steps that come just before and after too, to great a sort of weighted average.
        for idx in range(len(point_cloud_arr)):
            bin_number = np.round(point_cloud_arr[idx][2] / stride)
            freq_dict[bin_number] += 1
            freq_dict[bin_number - 1] += 1
            freq_dict[bin_number + 1] += 1
            freq_dict[bin_number - 2] += 1
            freq_dict[bin_number + 2] += 1
            freq_dict[bin_number - 2] += 1
            freq_dict[bin_number + 2] += 1
        # print("Aggregate dictionary")
        # print(freq_dict)
        max_key = max(freq_dict, key=freq_dict.get)
        # find the key or "discrete stride step" which has the maximum value.
        # We multiply the step by the stride to get the original z value.
        return max_key * stride

    # MODIFICATION END - HEURISTIC FUNCTIONS

    def __getitem__(self, index):

        rotate_to_center = cfg.DATA.RTC
        with_extra_feat = cfg.DATA.WITH_EXTRA_FEAT

        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        cls_type = self.type_list[index]
        # assert cls_type in KITTICategory.CLASSES, cls_type
        # size_class = KITTICategory.CLASSES.index(cls_type)

        assert cls_type in self.category_info.CLASSES, '%s not in category_info' % cls_type
        size_class = self.category_info.CLASSES.index(cls_type)

        # Compute one hot vector
        if self.one_hot:
            one_hot_vec = np.zeros((len(self.category_info.CLASSES)))
            one_hot_vec[size_class] = 1

        # Get point cloud
        if rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        if not with_extra_feat:
            point_set = point_set[:, :3]

        # MODIFICATION START - RUN HEURISTIC FUNCTION AND EVALUATE
        z_sliding_window_regress = self.regress_to_find_center_aggregate(point_set, 0.5)

        ### START CODE FOR EVALUATING HEURISTIC

        # self.errorMargins = np.append(self.errorMargins, np.absolute(z_sliding_window_regress - self.center_3d_raw_kitti[index][2]))
        
        # if (index > 0.999 * len(self.input_list)) and (self.printPercentiles is False):
        #     print('PERCENTILES = ' + str(np.percentile(self.errorMargins, [5*x for x in range(0, 21)])))
        #     self.printPercentiles = True
        #     percentileFile = open('errorPercentiles', 'w+')
        #     percentileFile.write('PERCENTILES FOR [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] = ' + str(np.percentile(self.errorMargins, [5*x for x in range(0, 21)])))
        #     percentileFile.close()
        #     # plt.plot(errorBuckets, self.errorMargins)
        #     # plt.show()

        ### END CODE FOR EVALUATING HEURISTIC

        # MODIFICATION END - RUN HEURISTIC FUNCTION AND EVALUATE

        # Resample
        if self.npoints > 0:
            # choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            choice = np.random.choice(point_set.shape[0], self.npoints, point_set.shape[0] < self.npoints)

        else:
            choice = np.random.permutation(len(point_set.shape[0]))

        point_set = point_set[choice, :]

        box = self.box2d_list[index]
        P = self.calib_list[index]['P2'].reshape(3, 4)

        # MODIFICATION START - PASS GENERATED Z VALUES TO GENERATE REF
        ref1, ref2, ref3, ref4 = self.generate_ref(box, P, z_sliding_window_regress)
        # MODIFICATION END - PASS GENERATED Z VALUES TO GENERATE REF

        if rotate_to_center:
            ref1 = self.get_center_view(ref1, index)
            ref2 = self.get_center_view(ref2, index)
            ref3 = self.get_center_view(ref3, index)
            ref4 = self.get_center_view(ref4, index)

        if self.from_rgb_detection:

            data_inputs = {
                'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
                'rot_angle': torch.FloatTensor([rot_angle]),
                'rgb_prob': torch.FloatTensor([self.prob_list[index]]),
                'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
                'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
                'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
                'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),

            }

            if not rotate_to_center:
                data_inputs.update({'rot_angle': torch.zeros(1)})

            if self.one_hot:
                data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

            return data_inputs

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index].astype(np.int64)
        seg = seg[choice]

        # Get center point of 3D box
        if rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        box3d_size = self.size_list[index]

        # Size
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

                ref1[:, 0] *= -1
                ref2[:, 0] *= -1
                ref3[:, 0] *= -1
                ref4[:, 0] *= -1

        if self.random_shift:
            max_depth = cfg.DATA.MAX_DEPTH
            l, w, h = self.size_list[index]
            dist = np.sqrt(np.sum(l ** 2 + w ** 2))
            shift = np.clip(np.random.randn() * dist * 0.2, -0.5 * dist, 0.5 * dist)
            shift = np.clip(shift + box3d_center[2], 0, max_depth) - box3d_center[2]
            point_set[:, 2] += shift
            box3d_center[2] += shift

        labels_ref2 = self.generate_labels(box3d_center, box3d_size, heading_angle, ref2, P)

        data_inputs = {
            'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
            'rot_angle': torch.FloatTensor([rot_angle]),
            'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
            'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
            'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
            'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),

            'cls_label': torch.LongTensor(labels_ref2),
            'box3d_center': torch.FloatTensor(box3d_center),
            'box3d_heading': torch.FloatTensor([heading_angle]),
            'box3d_size': torch.FloatTensor(box3d_size),
            'size_class': torch.LongTensor([size_class]),
            'seg_label': torch.LongTensor(seg.astype(np.int64))
        }

        if not rotate_to_center:
            data_inputs.update({'rot_angle': torch.zeros(1)})

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        return data_inputs

    def generate_labels(self, center, dimension, angle, ref_xyz, P):
        box_corner1 = compute_box_3d(center, dimension * 0.5, angle)
        box_corner2 = compute_box_3d(center, dimension, angle)

        labels = np.zeros(len(ref_xyz))
        inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)
        inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)

        labels[inside2] = -1
        labels[inside1] = 1
        # dis = np.sqrt(((ref_xyz - center)**2).sum(1))
        # print(dis.min())
        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1

        return labels

    # MODIFICATION START - RETURN DEPTH BOUNDS TO SEARCH
    def resolve_centers_z(self, z_sliding_window_regress):
        lower = max(0, z_sliding_window_regress - 8)
        upper = min(70, z_sliding_window_regress + 8)

        if z_sliding_window_regress < 8:
            upper += 8 - z_sliding_window_regress

        if z_sliding_window_regress > 62:
            lower -= z_sliding_window_regress - 62
        return lower, upper
    # MODIFICATION END - RETURN DEPTH BOUNDS TO SEARCH

    def generate_ref(self, box, P, z_sliding_window_regress):

        s1, s2, s3, s4 = cfg.DATA.STRIDE
        max_depth = cfg.DATA.MAX_DEPTH
        # MODIFIATION START - GENERATE DEPTH BOUNDS FOR SEARCHING
        z_sliding_window_regress = np.floor(z_sliding_window_regress)
        lower, upper = self.resolve_centers_z(z_sliding_window_regress)  # Modified z axis bounds
        # MODIFICATION END - GENERATE DEPTH BOUNDS FOR SEARCHING

        # MODIFICATION START - SEARCH THROUGH GENERATED DEPTH BOUNDS INSTEAD OF 70M
        z1 = np.arange(lower, upper, s1) + s1 / 2.
        z2 = np.arange(lower, upper, s2) + s2 / 2.
        z3 = np.arange(lower, upper, s3) + s3 / 2.
        z4 = np.arange(lower, upper, s4) + s4 / 2.
        # MODIFICATION END - SEARCH THROUGH GENERATED DEPTH BOUNDS INSTEAD OF 70M

        cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,

        xyz1 = np.zeros((len(z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = z1
        xyz1_rect = project_image_to_rect(xyz1, P)

        xyz2 = np.zeros((len(z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = z2
        xyz2_rect = project_image_to_rect(xyz2, P)

        xyz3 = np.zeros((len(z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = z3
        xyz3_rect = project_image_to_rect(xyz3, P)

        xyz4 = np.zeros((len(z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = z4
        xyz4_rect = project_image_to_rect(xyz4, P)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0),
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view,
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))

    def get_center_view(self, point_set, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(point_set)
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry

def compute_alpha(x, z, ry):

    beta = np.arctan2(z, x)
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry

    return alpha

def collate_fn(batch):
    return default_collate(batch)


if __name__ == '__main__':

    cfg.DATA.DATA_ROOT = 'kitti/data/pickle_data'
    cfg.DATA.RTC = True
    dataset = ProviderDataset(1024, split='val', random_flip=True, one_hot=True, random_shift=True)

    for i in range(len(dataset)):
        data = dataset[i]

        for name, value in data.items():
            print(name, value.shape)

        input()

    '''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    tic = time.time()
    for i, data_dict in enumerate(train_loader):

        # for key, value in data_dict.items():
        #     print(key, value.shape)

        print(time.time() - tic)
        tic = time.time()

        # input()
    '''

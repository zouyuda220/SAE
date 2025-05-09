import glob
import os
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PointShift_Dataset_UCF_QNRF(Dataset):

    def __init__(self, root_path, phase='train'):

        self.root_path = root_path
        self.phase = phase

        self.img_paths = glob.glob(self.root_path + '/*.jpg')
        self.img_paths.sort()
        # print(self.img_paths)
        self.n_samples = len(self.img_paths)
        numbers = 0
        self.images = []
        self.x_y_distance = []
        for img_path in self.img_paths:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            gt_path = img_path.replace('Train', 'Train_x_y_0.4_closest_distance').replace('jpg', 'npy')
            gt = np.load(gt_path)
            numbers = max(numbers, gt.shape[0])
            self.images.append(img)
            self.x_y_distance.append(gt)

    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = os.path.basename(self.img_paths[index])
        x_y_distance = self.x_y_distance[index]
        img = self.images[index]

        if self.phase == 'train':
            scale_range = [0.7, 1.3]
            scale = random.uniform(*scale_range)
            if min(img.shape[:2]) * scale < 520:
                scale = 520 / min(img.shape[:2])

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            x_y_distance = x_y_distance * scale

        if self.phase == 'train':
            height, width = img.shape[:2]

            height_start = random.randint(0, height - 512)
            height_end = height_start + 512
            width_start = random.randint(0, width - 512)
            width_end = width_start + 512

            mask_x_y_distance = (height_start <= x_y_distance[:, 1]) & (x_y_distance[:, 1] <= height_end - 1) & (width_start <= x_y_distance[:, 0]) & (x_y_distance[:, 0] <= width_end - 1)
            x_y_distance = x_y_distance[mask_x_y_distance]
            x_y_distance[:, 0] = x_y_distance[:, 0] - width_start
            x_y_distance[:, 1] = x_y_distance[:, 1] - height_start

            img = img[height_start:height_end, width_start:width_end]

        if self.phase == 'train' and random.randint(0, 1) == 1:
            img = img[:, ::-1, :].copy() 
            x_y_distance[:, 0] = img.shape[1] - 1 - x_y_distance[:, 0]

        img_view = img.copy()
        img = img.transpose((2, 0, 1))  # convert the order to (channel, heights, widths)
        img = img / 255
        img_tensor = torch.tensor(img, dtype=torch.float)
        img_tensor = transforms.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.phase == 'test':
            height, width = img_tensor.shape[1:]
            points_x = np.round(np.clip(x_y_distance[:, 0], 0, width - 1)).astype(np.int32)  # omit the decimal, which is similar to int()
            points_y = np.round(np.clip(x_y_distance[:, 1], 0, height - 1)).astype(np.int32)
            crop_number_row = (img_tensor.shape[-2] - 1) // 1024 + 1
            crop_number_col = (img_tensor.shape[-1] - 1) // 1024 + 1
            target = [{} for i in range(crop_number_row * crop_number_col)]

            pad_img_tensor = torch.nn.functional.pad(img_tensor, (0, crop_number_col * 1024 - img.shape[-1], 0, crop_number_row * 1024 - img.shape[-2]), mode='constant', value=0.0)
            new_img_tensor = torch.ones((crop_number_row * crop_number_col, 3, 1024, 1024))

            for row in range(crop_number_row):
                for col in range(crop_number_col):
                    new_img_tensor[col + row * crop_number_col] = pad_img_tensor[:, 1024 * row: (1024 * row + 1024), 1024 * col: (1024 * col + 1024)]

                    mask_x_y_distance_patch = (1024 * row <= x_y_distance[:, 1]) & (x_y_distance[:, 1] <= (1024 * row + 1024 - 1)) & (1024 * col <= x_y_distance[:, 0]) & (x_y_distance[:, 0] <= (1024 * col + 1024 - 1))
                    points_number_patch = np.sum(mask_x_y_distance_patch)
                    vis = [True] * points_number_patch + [False] * (15000 - points_number_patch)
                    vis = np.array(vis)
                    target[col + row * crop_number_col]['vis'] = vis
                    target[col + row * crop_number_col]['points_x'] = np.zeros(15000).astype(np.int32)
                    target[col + row * crop_number_col]['points_y'] = np.zeros(15000).astype(np.int32)
                    target[col + row * crop_number_col]['distance'] = np.zeros(15000).astype(np.int32)

                    target[col + row * crop_number_col]['points_x'][:int(points_number_patch)] = np.round(x_y_distance[mask_x_y_distance_patch, 0]).astype(np.int32) - col * 1024
                    target[col + row * crop_number_col]['points_y'][:int(points_number_patch)] = np.round(x_y_distance[mask_x_y_distance_patch, 1]).astype(np.int32) - row * 1024
                    target[col + row * crop_number_col]['distance'][:int(points_number_patch)] = np.array(x_y_distance[mask_x_y_distance_patch, 2])
                    target[col + row * crop_number_col]['row'] = row
                    target[col + row * crop_number_col]['col'] = col
                    target[col + row * crop_number_col]['image_path'] = self.img_paths[index]
            img_tensor = new_img_tensor


        if self.phase == 'train':
            # radius = np.random.rand(img.shape[:2]) * max_dist
            radius = np.random.rand(*img.shape[1:])
            angle = 2 * np.pi * np.random.rand(*img.shape[1:])

            shift_x = radius * np.cos(angle) / 1.
            shift_y = radius * np.sin(angle) / 1.

            height, width = img.shape[1:]

            points_number = x_y_distance.shape[0]
            vis = [True] * points_number + [False] * (15000 - points_number)
            vis = np.array(vis)

            x_y_distance = np.concatenate((x_y_distance, np.zeros((15000 - x_y_distance.shape[0], 3))), axis=0)

            # anchor_x, anchor_y = self.anchor(max_dist)
            points_x = np.round(np.clip(x_y_distance[:, 0], 0, width - 1)).astype(np.int32)  # omit the decimal, which is similar to int()

            points_y = np.round(np.clip(x_y_distance[:, 1], 0, height - 1)).astype(np.int32)

            cur_shift_x = shift_x[points_y, points_x] * x_y_distance[:, 2]  # noise alone the x axis for point in points shape[n]
            cur_shift_y = shift_y[points_y, points_x] * x_y_distance[:, 2]  # noise alone the y axis for point in points shape[n]

            shifted_x = x_y_distance[:, 0] + cur_shift_x  # x coordinate of noising points shape[n]
            shifted_y = x_y_distance[:, 1] + cur_shift_y  # y coordinate of noising points shape[n]

            shifted_x = np.clip(shifted_x, 0, width - 1)
            shifted_y = np.clip(shifted_y, 0, height - 1)

            shifted_x_int = np.round(shifted_x).astype(np.int32)  # int(x) coordinate of noising points shape[n]
            shifted_y_int = np.round(shifted_y).astype(np.int32)  # int(y) coordinate of noising points shape[n]


            target_dict = {'vis': vis, 'raw_img': img_view, 'shifted_x': shifted_x_int,
                        'shifted_y': shifted_y_int, 'image_path': self.img_paths[index], "points_x": points_x, 'points_y': points_y}
            target = target_dict

        return img_tensor, target



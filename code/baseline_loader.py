import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from torchvision import transforms


class NYU_v2(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode='train1', crop_h=None, crop_w=None, opt=None):
        json_file = os.path.join(dataroot, 'nyu_v2_3task.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.triples = info[mode]
        self.opt = opt
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 480
            self.crop_w = 640
        self.mode = mode
        self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))

    def __len__(self):
        return len(self.triples)
        # return 6

    @staticmethod
    def __scale__(img, label1, label2, label3):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        label1 = np.expand_dims(cv2.resize(label1, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label2 = cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        label3 = np.expand_dims(cv2.resize(label3, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)

        return img_new, label1, label2, label3

    @staticmethod
    def __mirror__(img, label1, label2, label3):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            label1 = label1[:, ::-1]
            label2 = label2[:, ::-1]
            label3 = label3[:, ::-1]
        return img, label1, label2, label3

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img, label1, label2, label3, crop_h, crop_w, ignore_label=255):
        # combining
        label = np.concatenate((label1, label2), axis=2).astype('float32')
        label -= ignore_label
        combined = np.concatenate((img, label, label3), axis=2)
        image_shape = img.shape
        label3_shape = label3.shape
        # padding to the crop size
        pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # separating
        img_cdim = image_shape[-1]
        label1_cdim = label1.shape[-1]
        label3_cdim = label3_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        label3_crop = deepcopy(combined_crop[:, :, -label3_cdim:])
        label_crop = combined_crop[:, :, img_cdim: -label3_cdim]
        label_crop = (label_crop + ignore_label).astype('uint8')
        label1_crop = label_crop[:, :, :label1_cdim]
        label2_crop = label_crop[:, :, label1_cdim:]
        return img_crop, label1_crop, label2_crop, label3_crop

    def __getitem__(self, item):
        img_path, seg_path, normal_path, depth_path = self.triples[item]
        img = cv2.imread(os.path.join(self.dataroot, img_path))
        seg = np.expand_dims(cv2.imread(os.path.join(self.dataroot, seg_path), cv2.IMREAD_GRAYSCALE), axis=-1)
        normal = cv2.imread(os.path.join(self.dataroot, normal_path))
        depth = np.expand_dims(np.load(os.path.join(self.dataroot, depth_path)), axis=-1)
        if self.mode in ['train', 'train1', 'train2']:
            img, seg, normal, depth = self.__scale__(img, seg, normal, depth)
            img, seg, normal, depth = self.__mirror__(img, seg, normal, depth)
            img, seg, normal, depth = self.__random_crop_and_pad_image_and_labels__(img, seg, normal, depth, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN

        return self.transform(img).float(), {'segment_semantic': torch.from_numpy(seg).permute(2, 0, 1),
                                             'depth_zbuffer': torch.from_numpy(depth).permute(2, 0, 1),
                                             'normal': torch.from_numpy(normal).permute(2, 0, 1)}


class CityScapes(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None, num_class=19, small_res=False, opt=None):
        json_file = os.path.join(dataroot, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.opt = opt
        self.groups = info[mode]
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            if small_res:
                self.crop_h = 128
                self.crop_w = 256
            else:
                self.crop_h = 256
                self.crop_w = 512
        self.mode = mode
        # self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.num_class = num_class

    def __len__(self):
        return len(self.groups)
        # return 16
        # return 6

    @staticmethod
    def __scale__(img, depth, label2, label7, label19):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        depth = np.expand_dims(cv2.resize(depth, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label2 = np.expand_dims(cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label7 = np.expand_dims(cv2.resize(label7, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label19 = np.expand_dims(cv2.resize(label19, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, depth, label2, label7, label19

    @staticmethod
    def __mirror__(img, depth, label2, label7, label19):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            depth = depth[:, ::-1]
            label2 = label2[:, ::-1]
            label7 = label7[:, ::-1]
            label19 = label19[:, ::-1]
        return img, depth, label2, label7, label19

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, crop_h, crop_w, ignore_label=-1.0):
        # combining
        label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
        label -= ignore_label
        combined = np.concatenate((img, depth, label), axis=2)
        image_shape = img.shape
        depth_shape = depth.shape
        # padding to the crop size
        pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # separating
        img_cdim = image_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        depth_cdim = depth_shape[-1]
        depth_crop = deepcopy(combined_crop[:, :, img_cdim: img_cdim + depth_cdim]).astype('float')
        label_crop = combined_crop[:, :, img_cdim + depth_cdim:]
        label_crop = label_crop + ignore_label
        label2_crop = np.expand_dims(label_crop[:, :, 0].astype('int'), axis=-1)
        label7_crop = np.expand_dims(label_crop[:, :, 1].astype('int'), axis=-1)
        label19_crop = np.expand_dims(label_crop[:, :, 2].astype('int'), axis=-1)

        return img_crop, depth_crop, label2_crop, label7_crop, label19_crop

    def __getitem__(self, item):
        # TODO RGB -> BGR
        img_path, depth_path, label2_path, label7_path, label19_path = self.groups[item]
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        depth = np.load(os.path.join(self.dataroot, depth_path))
        label2 = np.expand_dims(np.load(os.path.join(self.dataroot, label2_path)), axis=-1)
        label7 = np.expand_dims(np.load(os.path.join(self.dataroot, label7_path)), axis=-1)
        label19 = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
        if self.mode in ['train', 'train1', 'train2']:
            img, depth, label2, label7, label19 = self.__scale__(img, depth, label2, label7, label19)
            img, depth, label2, label7, label19 = self.__mirror__(img, depth, label2, label7, label19)
            img, depth, label2, label7, label19 = self.__random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN

        if self.num_class == 2:
            seg = torch.from_numpy(label2).permute(2, 0, 1)
        elif self.num_class == 7:
            seg = torch.from_numpy(label7).permute(2, 0, 1)
        elif self.num_class == 19:
            seg = torch.from_numpy(label19).permute(2, 0, 1)
        elif self.num_class == -1:
            seg = torch.from_numpy(label19).permute(2, 0, 1)
        else:
            raise ValueError('%d class is not supported in Cityscapes' % self.num_class)

        return torch.from_numpy(img).permute(2, 0, 1).float(), {'segment_semantic': seg,
                                                                'depth_zbuffer': torch.from_numpy(depth).permute(2, 0, 1)}




class CityScapesNew(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None, num_class=19, small_res=False, opt=None):
        json_file = os.path.join(dataroot, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.opt = opt
        self.groups = info[mode]
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            if small_res:
                self.crop_h = 128
                self.crop_w = 256
            else:
                self.crop_h = 256
                self.crop_w = 512
        self.mode = mode
        # self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.num_class = num_class

    def __len__(self):
        return len(self.groups)
        # return 16
        # return 6

    @staticmethod
    def __scale__(img, depth, label2, label7, label19):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        depth = np.expand_dims(cv2.resize(depth, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label2 = np.expand_dims(cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label7 = np.expand_dims(cv2.resize(label7, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label19 = np.expand_dims(cv2.resize(label19, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, depth, label2, label7, label19

    @staticmethod
    def __mirror__(img, depth, label2, label7, label19):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            depth = depth[:, ::-1]
            label2 = label2[:, ::-1]
            label7 = label7[:, ::-1]
            label19 = label19[:, ::-1]
        return img, depth, label2, label7, label19

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, crop_h, crop_w, ignore_label=-1.0):
        # combining
        label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
        label -= ignore_label
        combined = np.concatenate((img, depth, label), axis=2)
        image_shape = img.shape
        depth_shape = depth.shape
        # padding to the crop size
        pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # separating
        img_cdim = image_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        depth_cdim = depth_shape[-1]
        depth_crop = deepcopy(combined_crop[:, :, img_cdim: img_cdim + depth_cdim]).astype('float')
        label_crop = combined_crop[:, :, img_cdim + depth_cdim:]
        label_crop = label_crop + ignore_label
        label2_crop = np.expand_dims(label_crop[:, :, 0].astype('int'), axis=-1)
        label7_crop = np.expand_dims(label_crop[:, :, 1].astype('int'), axis=-1)
        label19_crop = np.expand_dims(label_crop[:, :, 2].astype('int'), axis=-1)

        return img_crop, depth_crop, label2_crop, label7_crop, label19_crop

    def __getitem__(self, item):
        # TODO RGB -> BGR
        img_path, depth_path, label2_path, label7_path, label19_path = self.groups[item]
        # print(img_path, '\n\n\n')
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        original_img = np.load(os.path.join(self.dataroot, img_path))
        depth = np.load(os.path.join(self.dataroot, depth_path))
        label2 = np.expand_dims(np.load(os.path.join(self.dataroot, label2_path)), axis=-1)
        label7 = np.expand_dims(np.load(os.path.join(self.dataroot, label7_path)), axis=-1)
        label19 = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
        if self.mode in ['train', 'train1', 'train2']:
            img, depth, label2, label7, label19 = self.__scale__(img, depth, label2, label7, label19)
            img, depth, label2, label7, label19 = self.__mirror__(img, depth, label2, label7, label19)
            img, depth, label2, label7, label19 = self.__random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN

        if self.num_class == 2:
            seg = torch.from_numpy(label2).permute(2, 0, 1)
        elif self.num_class == 7:
            seg = torch.from_numpy(label7).permute(2, 0, 1)
        elif self.num_class == 19:
            seg = torch.from_numpy(label19).permute(2, 0, 1)
        elif self.num_class == -1:
            seg = torch.from_numpy(label19).permute(2, 0, 1)
        else:
            raise ValueError('%d class is not supported in Cityscapes' % self.num_class)

        return original_img, torch.from_numpy(img).permute(2, 0, 1).float(), {'segment_semantic': seg,
                                                                              'depth_zbuffer': torch.from_numpy(depth).permute(2, 0, 1)}

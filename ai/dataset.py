import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from util import *

import cairosvg
from PIL import Image

## 손상된 PNG 파일 개수 저장


## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None, mode=None):
        corrupted_png_count = 0
        corrupted_png_files = []

        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        self.mode = mode

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        # 🔹 모든 .jpg 파일 리스트 가져오기
        if self.mode == "train":
            lst_input = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        else:
            lst_input = [f for f in os.listdir(self.data_dir) if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
        
        self.lst_input = sorted(lst_input)

        if self.mode == "test":
            self.lst_label = None
        else:
            self.lst_label = []
            for f in self.lst_input:
                label_name = f.replace('.jpg', '_close_wall_inverted.png')  # 🔹 라벨 이름 생성
                label_path = os.path.join(self.data_dir, label_name)
                if os.path.exists(label_path):  # 🔹 라벨 파일이 존재하면 추가
                    self.lst_label.append(label_name)
                else:
                    print(f"Warning: {label_name} not found!")  # 라벨 없는 경우 경고

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        input_path = os.path.join(self.data_dir, self.lst_input[index])
        input_img = plt.imread(input_path)

        # 흑백 이미지 처리 (3채널 변환)
        if input_img.ndim == 2:
            input_img = np.stack([input_img] * 3, axis=-1)

        # 정규화 (0~1)
        if input_img.dtype == np.uint8:
            input_img = input_img / 255.0

        data = {'input': input_img}  # 🔹 테스트 모드에서는 label 없이 input만 사용

        if self.mode != "test":  # 🔹 학습 모드일 때만 label을 로드
            label_path = os.path.join(self.data_dir, self.lst_label[index])
            
            label_img = plt.imread(label_path)
        
            if label_img.ndim == 2:
                label_img = np.stack([label_img] * 3, axis=-1)

            if label_img.dtype == np.uint8:
                label_img = label_img / 255.0

            if label_img.shape[-1] == 4:
                label_img = label_img[:, :, :3]

            data['label'] = label_img

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)
        return data

    def to_tensor(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)  # HWC -> CHW 변환
            data[key] = torch.from_numpy(value)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        # Updated at Apr 5 2020
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # input = (input - self.mean) / self.std
        # label = (label - self.mean) / self.std
        #
        # data = {'label': label, 'input': input}

        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        # data = {'label': label, 'input': input}

        return data


class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1],
                                                    self.shape[2]))

        return data
"""
about dataset
                        README

This file gives documentation for the cars 196 dataset.
(http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

----------------------------------------
Metadata/Annotations
----------------------------------------
Descriptions of the files are as follows:

-cars_meta.mat:
  Contains a cell array of class names, one for each class.

-cars_train_annos.mat:
  Contains the variable 'annotations', which is a struct array of length
  num_images and where each element has the fields:
    bbox_x1: Min x-value of the bounding box, in pixels
    bbox_x2: Max x-value of the bounding box, in pixels
    bbox_y1: Min y-value of the bounding box, in pixels
    bbox_y2: Max y-value of the bounding box, in pixels
    class: Integral id of the class the image belongs to.
    fname: Filename of the image within the folder of images.

-cars_test_annos.mat:
  Same format as 'cars_train_annos.mat', except the class is not provided.
"""

import os
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image


class StanfordCarsDataSet(Dataset):
    """base class"""

    def __init__(self, dir_name, data_frame, transforms=None):
        self.dir_name = dir_name
        # annotations
        self.cars_meta = loadmat(os.path.join(os.path.join(dir_name, 'car_devkit/devkit'),
                                              'cars_meta.mat'))
        self.cars_train_annos = loadmat(os.path.join(os.path.join(dir_name, 'cars_train/cars_train'),
                                                     'cars_train_annos.mat'))
        self.cars_test_annos = loadmat(os.path.join(os.path.join(dir_name, 'cars_test/cars_test'),
                                                    'cars_test_annos.mat'))
        self.transforms = transforms
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)


class StanfordTrainDataSet(StanfordCarsDataSet):
    """ train dataset"""
    def __init__(self, dir_name, data_frame, transforms):
        super().__init__(dir_name, data_frame, transforms)
        self.le = LabelEncoder()
        self.data_frame = self._create_train_df()

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_name, self.data_frame.iloc[idx, 5])
        img = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 6]
        bbox = list(self.data_frame.iloc[idx, :4])
        bbox = torch.from_numpy(np.array(bbox, dtype=np.uint8))
        if self.transforms:
            img = self.transforms(img)
        return img, label, bbox

    def _get_labels(self):
        """
        :return: labels DataFrame
        """
        labels = [c for c in self.cars_meta['class_names'][0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        return labels

    def _create_train_df(self):
        """
        create a Dataframe for train data where:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 - coords of bounding box e.g. x_min, y_min, x_max, y_max
        class - name of class
        fname - path to image
        label - correct class value
        :return: DataFrame
        """
        # create dataframe
        frame = [[i.flat[0] for i in line] for line in self.cars_train_annos['annotations'][0]]
        df_train = pd.DataFrame(frame, columns=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname'])
        df_train['class'] = df_train['class'] - 1
        df_train['fname'] = [os.path.join(os.path.join(
            self.dir_name, 'cars_train/cars_train'), f) for f in df_train['fname']]
        # merge dataframe and labels
        labels = self._get_labels()
        train_data_df = df_train.merge(labels, left_on='class', right_index=True)
        train_data_df = df_train.sort_index()
        return train_data_df

    def cls_names_to_numeric(self):
        """ convert class names of data to numeric """
        self.data_frame['labels'] = self.le.fit_transform(self.data_frame.labels.values)

    def get_class_names(self):
        """
        :return: list of class names
        """
        return list(self.le.classes_)


class StanfordTestDataSet(StanfordCarsDataSet):
    """ test dataset"""
    def __init__(self, dir_name, data_frame, transforms):
        super().__init__(dir_name, data_frame, transforms)
        self.data_frame = self._create_test_df()

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir_name, self.data_frame.iloc[idx, -1])
        img = Image.open(img_name).convert('RGB')

        bbox = list(self.data_frame.iloc[idx, :4])
        bbox = torch.from_numpy(np.array(bbox, dtype=np.uint8))

        if self.transforms:
            img = self.transforms(img)

        return img, bbox

    def _create_test_df(self):
        """
        create a DataFrame for test data where:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 - coords of bounding box e.g. x_min, y_min, x_max, y_max
        fname - path to image
        :return: DataFrame
        """
        frame = [[i.flat[0] for i in line] for line in self.cars_test_annos['annotations'][0]]
        df_test = pd.DataFrame(frame, columns=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname'])
        df_test['fname'] = [os.path.join(
            os.path.join(self.dir_name, 'cars_test/cars_test'), f) for f in df_test['fname']]
        return df_test
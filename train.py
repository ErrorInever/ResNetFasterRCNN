import torch
import torchvision
import numpy as np
import random

from model.cfg import cfg
from torchvision import transforms
from data.dataset import StanfordTrainDataSet, StanfordTestDataSet
from torch.utils.data import DataLoader


def set_seed(val):
    """freeze random sequences"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    dir_name = input(str('input dir name where data stored'))
    set_seed(cfg.RANDOM_SEED)
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type is not 'cuda':
        print("WARNING: cuda is not available")

    # define transforms for images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD)
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD)
    ])

    # prepare datasets
    train_dataset = StanfordTrainDataSet(dir_name, transforms=train_transform)
    test_dataset = StanfordTestDataSet(dir_name, transforms=val_test_transforms)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=5)
    print("info: \ntrain data [batches:{}, files:{}]\n"
          "test data [batches:{}, files:{}]\n".format(len(train_dataloader), len(train_dataset),
                                                      len(test_dataloader), len(test_dataset)))


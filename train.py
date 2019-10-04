import torch
import torchvision
import numpy as np
import random

from model.cfg import cfg
from torchvision import transforms
from data.dataset import StanfordTrainDataSet, StanfordTestDataSet
from torch.utils.data import DataLoader
from model import faster_rcnn


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
        print("WARNING: cuda is not available, using cpu")

    # define transforms for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.TRANSFORM.MEAN, cfg.TRANSFORM.STD)
    ])

    # prepare datasets
    train_dataset = StanfordTrainDataSet(dir_name, transforms=transform)
    test_dataset = StanfordTestDataSet(dir_name, transforms=transform)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    print("info: \ntrain data [batches:{}, files:{}]\n"
          "test data [batches:{}, files:{}]\n".format(len(train_dataloader), len(train_dataset),
                                                      len(test_dataloader), len(test_dataset)))

    # define model
    model = faster_rcnn.resnet_50(num_classes=196)




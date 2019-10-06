import math
import sys

import torch
import numpy as np
import random
from model.cfg import cfg
from torchvision import transforms
from data.dataset import StanfordTrainDataSet, StanfordTestDataSet
from torch.utils.data import DataLoader
from model import faster_rcnn
from data.collate import collate
from vision.references.detection import utils


def set_seed(val):
    """freeze random sequences"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    based on vision/references/detection/train/train_one_epoch
    :param model: object nn.Module of net
    :param optimizer: object optimizer
    :param data_loader: object DataLoader
    :param device: only cuda
    :param epoch: number of epoch
    :return:
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # all turn to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward, get dictionary
        # {loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg}
        loss_dict = model(images, targets)
        # sum of losses
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # set all gradient by zero
        optimizer.zero_grad()
        # backward though Net
        losses.backward()
        # gradient step
        optimizer.step()
        # reduce gradient step
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


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
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True, num_workers=4, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                 shuffle=True, num_workers=4)
    print("info: \ntrain data [batches:{}, files:{}]\n"
          "test data [batches:{}, files:{}]\n".format(len(train_dataloader), len(train_dataset),
                                                      len(test_dataloader), len(test_dataset)))

    # define model and params
    model = faster_rcnn.resnet_50(num_classes=196)
    # construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.LEARNING_RATE,
                                momentum=cfg.TRAIN.SGD.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    # decreases a learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.TRAIN.SCHEDULER.STEP_SIZE,
                                                   gamma=cfg.TRAIN.SCHEDULER.GAMMA)

    train_one_epoch(model, optimizer, lr_scheduler, train_dataloader, device, cfg.TRAIN.EPOCHS, print_freq=10)
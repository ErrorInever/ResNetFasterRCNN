import model.cfg as cfg
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def resnet(n_classes, backbone_n='resnet50', pretrained_backbone=True):
    """
    Faster-RCNN with
    :param n_classes: number of classes of Fast-RCNN-Predictor
    :param backbone_n: name of backbone which will extract feature maps
    :param pretrained_backbone: if True, return pretrained backbone of resnet
    :return: instance of FasterRCNN
    """
    if backbone_n not in names:
        raise Exception('Wrong backbone name')

    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_n,
                                                                               pretrained=pretrained_backbone)

    # set out channels for FasterRCNN
    backbone.out_channels = 256
    # define custom anchors for RPN
    anchor_generator = AnchorGenerator(sizes=cfg.ANCHOR_SCALES, aspect_ratios=cfg.ANCHOR_RATIOS)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    # define model
    model = FasterRCNN(backbone,
                       num_classes=n_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def pretrained_resnet_50(n_classes=91):
    """
    :param n_classes: number of classes of Fast-RCNN-Predictor
    :return: returns the Faster-RCNN-Resnet-50 pre-trained on COCO train2017
    """
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=n_classes,
                                                                pretrained=True,
                                                                progress=True)


def __custom_resnet():
    # TODO release custom backbone for RCNN
    pass

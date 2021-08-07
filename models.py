import torch
import gc
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import timm
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval, DetBenchTrain
from effdet.efficientdet import HeadNet

def my_resnet_fpn_backbone(backbone_name, pretrained):
    backbone = timm.create_model(backbone_name, pretrained=pretrained)
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

def fasterrcnn_resnet_fpn(backbone_name='resnet152', progress=True, num_classes=91, pretrained=True, pretrained_backbone=True, **kwargs):
    if backbone_name == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif backbone_name in ['resnet101', 'resnet152']:
        backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone)
    else:
        backbone = my_resnet_fpn_backbone(backbone_name, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def get_effdet(backbone, num_classes=1, img_size=512, mode='train', pretrained=True, pretrained_backbone=False):
    if pretrained:
        pretrained_backbone = False
    if backbone == 'ed0':
        config = get_efficientdet_config('tf_efficientdet_d0')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d0-d92fd44f.pth')
    elif backbone == 'ed1':
        config = get_efficientdet_config('tf_efficientdet_d1')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d1-4c7ebaf2.pth')
    elif backbone == 'ed2':
        config = get_efficientdet_config('tf_efficientdet_d2')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d2-cb4ce77d.pth')
    elif backbone == 'ed3':
        config = get_efficientdet_config('tf_efficientdet_d3')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d3-b0ea2cbc.pth')
    elif backbone == 'ed4':
        config = get_efficientdet_config('tf_efficientdet_d4')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d4-5b370b7a.pth')
    elif backbone == 'ed5':
        config = get_efficientdet_config('tf_efficientdet_d5')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d5-ef44aea8.pth')
    elif backbone == 'ed6':
        config = get_efficientdet_config('tf_efficientdet_d6')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d6-51cb0132.pth')
    elif backbone == 'ed7':
        config = get_efficientdet_config('tf_efficientdet_d7')
        checkpoint = torch.load('effdet-pretrained/tf_efficientdet_d7-f05bf714.pth')
    else:
        raise ValueError("BACKBONE!!!")
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    if pretrained:
        model.load_state_dict(checkpoint)
        del checkpoint
        gc.collect()
    config.num_classes = num_classes
    config.image_size = img_size
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    if mode == 'train':
        return DetBenchTrain(model, config)
    else:
        return DetBenchEval(model, config)

def get_effdet_test(backbone, num_classes=1, img_size=512):
    if backbone == 'ed0':
        config = get_efficientdet_config('tf_efficientdet_d0')
    elif backbone == 'ed1':
        config = get_efficientdet_config('tf_efficientdet_d1')
    elif backbone == 'ed2':
        config = get_efficientdet_config('tf_efficientdet_d2')
    elif backbone == 'ed3':
        config = get_efficientdet_config('tf_efficientdet_d3')
    elif backbone == 'ed4':
        config = get_efficientdet_config('tf_efficientdet_d4')
    elif backbone == 'ed5':
        config = get_efficientdet_config('tf_efficientdet_d5')
    elif backbone == 'ed6':
        config = get_efficientdet_config('tf_efficientdet_d6')
    elif backbone == 'ed7':
        config = get_efficientdet_config('tf_efficientdet_d7')
    else:
        raise ValueError("BACKBONE!!!")
    model = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = num_classes
    config.image_size = img_size
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchEval(model, config)

def get_effdet_pseudo(backbone, num_classes=1, img_size=512):
    if backbone == 'ed0':
        config = get_efficientdet_config('tf_efficientdet_d0')
    elif backbone == 'ed1':
        config = get_efficientdet_config('tf_efficientdet_d1')
    elif backbone == 'ed2':
        config = get_efficientdet_config('tf_efficientdet_d2')
    elif backbone == 'ed3':
        config = get_efficientdet_config('tf_efficientdet_d3')
    elif backbone == 'ed4':
        config = get_efficientdet_config('tf_efficientdet_d4')
    elif backbone == 'ed5':
        config = get_efficientdet_config('tf_efficientdet_d5')
    elif backbone == 'ed6':
        config = get_efficientdet_config('tf_efficientdet_d6')
    elif backbone == 'ed7':
        config = get_efficientdet_config('tf_efficientdet_d7')
    else:
        raise ValueError("BACKBONE!!!")
    model = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = num_classes
    config.image_size = img_size
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(model, config)
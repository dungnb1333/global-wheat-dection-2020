## Global Wheat Detection - 1st place solution

![Alt text](./images/gwd2020.png?raw=true "Optional Title")

This repo contains the source code of the 1st place solution for [Global Wheat Detection Challenge](https://www.kaggle.com/c/global-wheat-detection). In this competition, you’ll detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. Below you can find a outline of how to reproduce my solution.

### Summary
* Custom mosaic data augmentation
* MixUp
* Heavy augmentation
* EfficientDet
* Faster RCNN FPN
* Ensemble multi-scale model: Weighted-Boxes-Fusion
* Test time augmentation(HorizontalFlip, VerticalFlip, Rotate90)
* Pseudo labeling

### Augmentations
* Custom mosaic augmentation
![Alt text](./images/mosaic.png?raw=true "Optional Title")
* MixUp
* Heavy augmentation: RandomCrop, HorizontalFlip, VerticalFlip, ToGray, AdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, CLAHE, Sharpen, Emboss, RandomBrightnessContrast, HueSaturationValue
2 examples mixup+mosaic+augmentation:
![Alt text](./images/mixup.png?raw=true "Optional Title")

### Requirements 
- Ubuntu 18.04 LTS
- CUDA 10.1
- CuDNN 7.5.1
- Python 3.7.6
- python packages
```
$ conda create -n wheat_env python=3.7.6
$ conda activate wheat_env
$ pip install -r requirements.txt
```
- [nvidia-apex 0.1](https://github.com/NVIDIA/apex)

## DATASET
- download competition dataset at [link](https://www.kaggle.com/c/global-wheat-detection/data) then extract to ./dataset folder
- [Wheat 2017](https://plantimages.nottingham.ac.uk/)
- [Spike-wheat dataset](https://sourceforge.net/projects/spike-dataset/)
- extract external dataset and label
```
$ cd dataset
$ unzip spike-wheat.zip
$ unzip wheat2017.zip
```
./dataset folder structure should be:
```
dataset
├── sample_submission.csv
├── test
│   ├── 2fd875eaa.jpg
│   ├── ...
├── train
│   ├── 00333207f.jpg
│   ├── ...
├── trainset.csv
├── wheat2017
│   ├── wheat2017_0001.jpg
│   ├── ...
├── wheat2017.csv
├── spike-wheat
│   ├── spike0000.jpg
│   ├── ...
├── spike-wheat.csv
```

### Model
* [EfficientDet-PyTorch](https://github.com/rwightman/efficientdet-pytorch) licensed under Apache 2.0, Copyright Ross Wightman
* [Faster RCNN FPN](https://github.com/pytorch/vision/tree/master/torchvision/models/detection) licensed under BSD 3-Clause
* 5 folds cross validation
* Optimizer: Adam with initial LR 5e-4 for EfficientDet and SGD with initial LR 5e-3 for Faster RCNN FPN
* LR scheduler: cosine-annealing
* Warm-up 20 epochs with trainset + wheat2017 dataset + spike wheat dataset -> train 80 epochs with trainset + wheat2017
* Pseudo labeling

### Train all models from scratch
- Train models
```
$ cd effdet-pretrained && bash download.sh && cd ..
$ python effdet_train.py --folds 0 1 2 3 4 --backbone ed7 --img-size 768 --batch-size 8 --workers 16 --use-amp True
$ python effdet_train.py --folds 1 3 --backbone ed7 --img-size 1024 --batch-size 4 --workers 16 --use-amp True
$ python effdet_train.py --folds 4 --backbone ed5 --img-size 512 --batch-size 20 --workers 16 --use-amp True
$ python effdet_train.py --folds 1 --backbone ed6 --img-size 640 --batch-size 12 --workers 16 --use-amp True
$ python faster_rcnn_fpn_train.py --folds 1 --backbone resnet152 --img-size 1024 --batch-size 20 --workers 16
```
- Evaluate models
```
python evaluate.py --folds 0 --network effdet --backbone ed7 --img-size 768 --batch-size 16 --workers 8
python evaluate.py --folds 1 --network fasterrcnn --backbone resnet152 --img-size 1024 --batch-size 16 --workers 8
```
### Performance
| Network                  | image-size | Fold | Valid AP |
| :------------------------| :----------|:-----|:---------|
| EfficientDet-D7          | 768        | 0    | 0.710    |
| EfficientDet-D7          | 768        | 1    | 0.716    |
| EfficientDet-D7          | 768        | 2    | 0.707    |
| EfficientDet-D7          | 768        | 3    | 0.716    |
| EfficientDet-D7          | 768        | 4    | 0.713    |
| EfficientDet-D7          | 1024       | 1    | 0.718    | 
| EfficientDet-D7          | 1024       | 3    | 0.720    | 
| EfficientDet-D5          | 512        | 4    | 0.702    | 
| EfficientDet-D6          | 640        | 1    | 0.716    | 
| FasterRCNN-FPN-resnet152 | 1024       | 1    | 0.695    |

### Pseudo labeling
- Base: EfficientDet-d6 image-size 640 Fold1 0.716 Valid AP
- Round1: Train EfficientDet-d6 10 epochs with trainset + hidden testset (output of ensembling), load weight from base checkpoint \
  Result: [old testset] 0.7719 Public LB/0.7175 Private LB and [new testset] 0.7633 Public LB/0.6787 Private LB
- Round2: Continue train EfficientDet-d6 6 epochs with trainset + hidden testset (output of pseudo labeling round1), load weight from pseudo labeling round1 checkpoint \
  Result: [old testset]0.7754 Public LB/0.7205 Private LB and [new testset]0.7656 Public LB/0.6897 Private LB

### Kaggle kernel
[Final submission](https://www.kaggle.com/nguyenbadung/gwd2020)
import os
import numpy as np 
import cv2
import random
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from albumentations import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def get_aug(aug):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['category_id']))

def bb_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    iou = interArea / float(boxAArea)
    return iou

class WheatDataset(Dataset):
    def __init__(self, df, img_size, mode='train', network='FasterRCNN', bbox_removal_threshold=0.25):
        super(WheatDataset,self).__init__()
        self.df = df
        self.image_ids = list(np.unique(self.df.image_id.values))
        self.img_size = img_size
        self.root_dir = 'dataset/train'
        self.w2017_ext_dir = 'dataset/wheat2017'
        self.spike_ext_dir = 'dataset/spike-wheat'
        assert mode in  ['train', 'valid']
        self.mode = mode
        assert network in ['FasterRCNN', 'EffDet']
        self.network = network
        self.bbox_removal_threshold = bbox_removal_threshold
        if self.mode == 'train':
            random.shuffle(self.image_ids)
        self.train_transforms = get_aug([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToGray(p=0.01),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            OneOf([
                CLAHE(),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.25),
            HueSaturationValue(p=0.25)
        ])
        self.resize_transforms = get_aug([
            Resize(height=self.img_size, width=self.img_size, interpolation=1, p=1)
        ])

    def __len__(self):
        return len(self.image_ids)

    def refine_boxes(self, boxes):
        result_boxes = []
        for box in boxes:
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            result_boxes.append(box)
        result_boxes = np.array(result_boxes)
        return result_boxes

    def resize_image(self, image, boxes):
        cats = np.ones(boxes.shape[0], dtype=int)
        annotations = {'image': image, 'bboxes': boxes, 'category_id': cats}
        augmented = self.resize_transforms(**annotations)
        image = augmented['image']
        boxes = np.array(augmented['bboxes'])
        return image, boxes

    def crop_image(self, image, boxes, xmin, ymin, xmax, ymax):
        image = image[ymin:ymax,xmin:xmax,:]
        cutout_box = [xmin, ymin, xmax, ymax]
        result_boxes = []
        for box in boxes:
            iou = bb_overlap(box, cutout_box)
            if iou > self.bbox_removal_threshold:
                result_boxes.append(box)
        if len(result_boxes) > 0:
            result_boxes = np.array(result_boxes, dtype=float)
            result_boxes[:,[0,2]] -= xmin
            result_boxes[:,[1,3]] -= ymin
            result_boxes[:,[0,2]] = result_boxes[:,[0,2]].clip(0, xmax-xmin)
            result_boxes[:,[1,3]] = result_boxes[:,[1,3]].clip(0, ymax-ymin)
        else:
            result_boxes = np.array([], dtype=float).reshape(0,4)
        return image, result_boxes
    
    def random_crop_resize(self, image, boxes, img_size=1024, p=0.5):
        if random.random() > p:
            new_img_size = random.randint(int(0.75*img_size), img_size)
            x = random.randint(0, img_size-new_img_size)
            y = random.randint(0, img_size-new_img_size)
            image, boxes = self.crop_image(image, boxes, x, y, x+new_img_size, y+new_img_size)
            return self.resize_image(image, boxes)
        else:
            if self.img_size != 1024:
                return self.resize_image(image, boxes)
            else:
                return image, boxes

    def load_image_and_boxes(self, image_id):
        tmp_df = self.df.loc[self.df['image_id']==image_id]
        source = np.unique(tmp_df.source.values)[0]
        if source == 'wheat2017':
            img_path = '{}/{}.jpg'.format(self.w2017_ext_dir, image_id)
        elif source == 'spike':
            img_path = '{}/{}.jpg'.format(self.spike_ext_dir, image_id)
        else:
            img_path = '{}/{}.jpg'.format(self.root_dir, image_id)
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img, dtype=np.uint8)

        boxes = []
        for _, row in tmp_df.iterrows():
            if row['isbox'] == False:
                continue
            boxes.append([float(row['xmin']),float(row['ymin']),float(row['xmax']),float(row['ymax'])])

        boxes = self.refine_boxes(boxes)
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
        return img, boxes, source
    
    def load_cutmix_image_and_boxes(self, image_id, imsize=1024):   #custom mosaic data augmentation
        image_ids = self.image_ids.copy()
        image_ids.remove(image_id)
        cutmix_image_ids = [image_id] + random.sample(image_ids, 3)
        result_image = np.full((imsize, imsize, 3), 1, dtype=np.uint8)
        result_boxes = []
        
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]
        for i, img_id in enumerate(cutmix_image_ids):
            image, boxes, source = self.load_image_and_boxes(img_id)
            if source == 'spike':
                height, width = image.shape[0:2]
                if i == 0 or i == 3:
                    image, boxes = self.crop_image(image, boxes, xmin=width-1024, ymin=0, xmax=width, ymax=1024)
                else:
                    image, boxes = self.crop_image(image, boxes, xmin=0, ymin=0, xmax=1024, ymax=1024)
            if i == 0:
                image, boxes = self.crop_image(image, boxes, imsize-xc, imsize-yc, imsize, imsize)
                result_image[0:yc, 0:xc,:] = image
                result_boxes.extend(boxes)
            elif i == 1:
                image, boxes = self.crop_image(image, boxes, 0, imsize-yc, imsize-xc, imsize)
                result_image[0:yc, xc:imsize, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[0,2]] += xc
                result_boxes.extend(boxes)
            elif i == 2:
                image, boxes = self.crop_image(image, boxes, 0, 0, imsize-xc, imsize-yc)
                result_image[yc:imsize, xc:imsize, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[0,2]] += xc
                    boxes[:,[1,3]] += yc
                result_boxes.extend(boxes)
            else:
                image, boxes = self.crop_image(image, boxes, imsize-xc, 0, imsize, imsize-yc)
                result_image[yc:imsize, 0:xc, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[1,3]] += yc
                result_boxes.extend(boxes)
            del image
            del boxes
        del cutmix_image_ids
        del image_ids
        if len(result_boxes) == 0:
            result_boxes = np.array([], dtype=float).reshape(0,4)
        else:
            result_boxes = np.vstack(result_boxes)
            result_boxes[:,[0,2]] = result_boxes[:,[0,2]].clip(0, imsize)
            result_boxes[:,[1,3]] = result_boxes[:,[1,3]].clip(0, imsize)
        return result_image, result_boxes
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.mode == 'train':
            while(True):
                if random.random() > 0.5:
                    image, boxes, source = self.load_image_and_boxes(image_id)
                    if source == 'spike':
                        height, width = image.shape[0:2]
                        if random.random() > 0.5:
                            image, boxes = self.crop_image(image, boxes, xmin=0, ymin=0, xmax=1024, ymax=1024)
                        else:
                            image, boxes = self.crop_image(image, boxes, xmin=width-1024, ymin=0, xmax=width, ymax=1024)
                else:
                    image, boxes = self.load_cutmix_image_and_boxes(image_id)

                image, boxes = self.random_crop_resize(image, boxes, p=0.5)
                if len(boxes) > 0:
                    cats = np.ones(boxes.shape[0], dtype=int)
                    annotations = {'image': image, 'bboxes': boxes, 'category_id': cats}
                    augmented = self.train_transforms(**annotations)
                    image = augmented['image']
                    boxes = np.array(augmented['bboxes'])
                    break
        else:
            image, boxes, _ = self.load_image_and_boxes(image_id)
            if self.img_size != 1024:
                image, boxes = self.resize_image(image, boxes)

        if self.network == 'EffDet':
            if boxes.shape[0] == 0:
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64)
                }
            else:
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
                target = {
                    'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                    'labels': torch.ones((boxes.shape[0],), dtype=torch.int64)
                }
        else:
            if boxes.shape[0] == 0:
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "area": torch.zeros(0, dtype=torch.float32),
                    "iscrowd": torch.zeros((0,), dtype=torch.int64)
                }
            else:
                target = {}
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
                target['area'] = torch.as_tensor(area, dtype=torch.float32)
                target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
        image = image.astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image).permute(2,0,1)
        return image, target

class WheatTestset(Dataset):
    def __init__(self, df, img_size, root_dir='dataset/train', shuffle=True):
        super(WheatTestset,self).__init__()
        self.df = df
        self.image_ids = list(np.unique(self.df.image_id.values))
        if shuffle:
            random.shuffle(self.image_ids)
        self.img_size = img_size
        self.root_dir = root_dir
        self.transforms = Resize(height=self.img_size, width=self.img_size, interpolation=1, p=1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = '{}/{}.jpg'.format(self.root_dir, image_id)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = self.transforms(image=img)['image']
        img = img.astype(np.float32)
        img /= 255.0
        img = torch.from_numpy(img).permute(2,0,1)

        return img, image_id

class WheatPseudoTestset(Dataset):
    def __init__(self, df, img_size, mode='train', bbox_removal_threshold=0.25):
        super(WheatPseudoTestset,self).__init__()
        self.df = df
        self.image_paths = list(np.unique(self.df.image_path.values))
        self.img_size = img_size
        assert mode in  ['train', 'valid']
        self.mode = mode
        self.bbox_removal_threshold = bbox_removal_threshold
        if self.mode == 'train':
            random.shuffle(self.image_paths)
        self.train_transforms = get_aug([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToGray(p=0.01),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            OneOf([
                CLAHE(),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.25),
            HueSaturationValue(p=0.25)
        ])
        self.resize_transforms = get_aug([
            Resize(height=self.img_size, width=self.img_size, interpolation=1, p=1)
        ])

    def __len__(self):
        return len(self.image_paths)

    def refine_boxes(self, boxes):
        result_boxes = []
        for box in boxes:
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            result_boxes.append(box)
        result_boxes = np.array(result_boxes)
        return result_boxes

    def resize_image(self, image, boxes):
        cats = np.ones(boxes.shape[0], dtype=int)
        annotations = {'image': image, 'bboxes': boxes, 'category_id': cats}
        augmented = self.resize_transforms(**annotations)
        image = augmented['image']
        boxes = np.array(augmented['bboxes'])
        return image, boxes

    def crop_image(self, image, boxes, xmin, ymin, xmax, ymax):
        image = image[ymin:ymax,xmin:xmax,:]
        cutout_box = [xmin, ymin, xmax, ymax]
        result_boxes = []
        for box in boxes:
            iou = bb_overlap(box, cutout_box)
            if iou > self.bbox_removal_threshold:
                result_boxes.append(box)
        if len(result_boxes) > 0:
            result_boxes = np.array(result_boxes, dtype=float)
            result_boxes[:,[0,2]] -= xmin
            result_boxes[:,[1,3]] -= ymin
            result_boxes[:,[0,2]] = result_boxes[:,[0,2]].clip(0, xmax-xmin)
            result_boxes[:,[1,3]] = result_boxes[:,[1,3]].clip(0, ymax-ymin)
        else:
            result_boxes = np.array([], dtype=float).reshape(0,4)
        return image, result_boxes
    
    def random_crop_resize(self, image, boxes, img_size=1024, p=0.5):
        if random.random() > p:
            new_img_size = random.randint(int(0.75*img_size), img_size)
            x = random.randint(0, img_size-new_img_size)
            y = random.randint(0, img_size-new_img_size)
            image, boxes = self.crop_image(image, boxes, x, y, x+new_img_size, y+new_img_size)
            return self.resize_image(image, boxes)
        else:
            if self.img_size != 1024:
                return self.resize_image(image, boxes)
            else:
                return image, boxes

    def load_image_and_boxes(self, image_path):
        tmp_df = self.df.loc[self.df['image_path']==image_path]

        img = Image.open(image_path)
        img = img.convert('RGB')
        img = np.array(img, dtype=np.uint8)

        boxes = []
        for _, row in tmp_df.iterrows():
            if row['isbox'] == False:
                continue
            boxes.append([float(row['xmin']),float(row['ymin']),float(row['xmax']),float(row['ymax'])])
        boxes = self.refine_boxes(boxes)

        if img.shape[0] != 1024 or img.shape[1] != 1024:
            augs = get_aug([
                Resize(height=1024, width=1024, interpolation=1, p=1)
            ])
            cats = np.ones(boxes.shape[0], dtype=int)
            annotations = {'image': img, 'bboxes': boxes, 'category_id': cats}
            augmented = augs(**annotations)
            img = augmented['image']
            boxes = np.array(augmented['bboxes'])

        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
        return img, boxes
    
    def load_cutmix_image_and_boxes(self, image_path, imsize=1024):     #custom mosaic data augmentation
        image_paths = self.image_paths.copy()
        image_paths.remove(image_path)
        cutmix_image_paths = [image_path] + random.sample(image_paths, 3)
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]
        result_image = np.full((imsize, imsize, 3), 1, dtype=np.uint8)
        result_boxes = []
        for i, img_path in enumerate(cutmix_image_paths):
            image, boxes = self.load_image_and_boxes(img_path)
            if i == 0:
                image, boxes = self.crop_image(image, boxes, imsize-xc, imsize-yc, imsize, imsize)
                result_image[0:yc, 0:xc,:] = image
                result_boxes.extend(boxes)
            elif i == 1:
                image, boxes = self.crop_image(image, boxes, 0, imsize-yc, imsize-xc, imsize)
                result_image[0:yc, xc:imsize, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[0,2]] += xc
                result_boxes.extend(boxes)
            elif i == 2:
                image, boxes = self.crop_image(image, boxes, 0, 0, imsize-xc, imsize-yc)
                result_image[yc:imsize, xc:imsize, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[0,2]] += xc
                    boxes[:,[1,3]] += yc
                result_boxes.extend(boxes)
            else:
                image, boxes = self.crop_image(image, boxes, imsize-xc, 0, imsize, imsize-yc)
                result_image[yc:imsize, 0:xc, :] = image
                if boxes.shape[0] > 0:
                    boxes[:,[1,3]] += yc
                result_boxes.extend(boxes)
            del image
            del boxes
        del cutmix_image_paths
        del image_paths
        if len(result_boxes) == 0:
            result_boxes = np.array([], dtype=float).reshape(0,4)
        else:
            result_boxes = np.vstack(result_boxes)
            result_boxes[:,[0,2]] = result_boxes[:,[0,2]].clip(0, imsize)
            result_boxes[:,[1,3]] = result_boxes[:,[1,3]].clip(0, imsize)
        return result_image, result_boxes
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        if self.mode == 'train':
            while(True):
                if random.random() > 0.5:
                    image, boxes = self.load_image_and_boxes(image_path)
                else:
                    image, boxes = self.load_cutmix_image_and_boxes(image_path)
                image, boxes = self.random_crop_resize(image, boxes, p=0.5)
                if len(boxes) > 0:
                    cats = np.ones(boxes.shape[0], dtype=int)
                    annotations = {'image': image, 'bboxes': boxes, 'category_id': cats}
                    augmented = self.train_transforms(**annotations)
                    image = augmented['image']
                    boxes = np.array(augmented['bboxes'])
                    break
        else:
            image, boxes = self.load_image_and_boxes(image_path)
            if self.img_size != 1024:
                image, boxes = self.resize_image(image, boxes)

        if boxes.shape[0] == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            }
        else:
            boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.ones((boxes.shape[0],), dtype=torch.int64)
            }

        image = image.astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image).permute(2,0,1)
        return image, target

class BaseWheatTTA:
    def augment(self, images):
        raise NotImplementedError

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, images):
        return list(image.flip(1) for image in images)

    def effdet_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return self.prepare_boxes(boxes)

class TTAVerticalFlip(BaseWheatTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, images):
        return list(image.flip(2) for image in images)

    def effdet_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes

class TTARotate90(BaseWheatTTA):
    def __init__(self, image_size):
        self.image_size = image_size
    
    def fasterrcnn_augment(self, images):
        return list(torch.rot90(image, 1, (1, 2)) for image in images)
    
    def effdet_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return self.prepare_boxes(res_boxes)

class TTACompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def fasterrcnn_augment(self, images):
        for transform in self.transforms:
            images = transform.fasterrcnn_augment(images)
        return images

    def effdet_augment(self, images):
        for transform in self.transforms:
            images = transform.effdet_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)
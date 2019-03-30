import torch
import torch.utils.data as data
import pandas as pd
import cv2
import numpy as np
import random
import os
#list=os.listdir(r'F:\pc\senceClassification\UCMerced_LandUse')
CLASS=['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
       'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
       'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot',
       'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)


    return image

def randomVerticleFlip(image,  u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)

    return image

def randomRotate90(image,  u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)


    return image
def default_loader_val(filename):
    img=cv2.imread(filename)
    path=filename.spilt('/')
    classname=path[4]
    print(classname)
    label=np.array([int(CLASS.index(classname))],np.int64)
    print(CLASS.index(classname))
    return  img,label

def default_loader(filename):
    img=cv2.imread(filename)
    img=cv2.resize(img,(224,224))
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img = randomShiftScaleRotate(img,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img = randomHorizontalFlip(img)
    img = randomVerticleFlip(img)
    img = randomRotate90(img)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    #img=np.transpose(img,[2,0,1])
    path=filename.split('/')
    classname=path[4]
    label=np.array([int(CLASS.index(classname))],np.int64)

    return  img,label

def default_loader_val(filename):
    img=cv2.imread(filename)
    img=cv2.resize(img,(224,224))
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    #img=np.transpose(img,[2,0,1])
    path=filename.split('/')
    classname=path[4]
    label=np.array([int(CLASS.index(classname))],np.int64)

    return  img,label

class ImageFolder(data.Dataset):

    def __init__(self, trainlist):
        table = pd.read_table(trainlist, header=None, sep=',')
        trainlist = table.values
        self.ids = trainlist
        self.loader = default_loader


    def __getitem__(self, index):
        id = self.ids[index][0]
        img,label = self.loader(id)
        img = torch.Tensor(img)
        label = torch.LongTensor(label)
        return img,label

    def __len__(self):
        return len(self.ids)

class ImageFolder_val(data.Dataset):

    def __init__(self, trainlist):
        table = pd.read_table(trainlist, header=None, sep=',')
        trainlist = table.values
        self.ids = trainlist
        self.loader = default_loader_val


    def __getitem__(self, index):
        id = self.ids[index][0]
        img,label = self.loader(id)
        img = torch.Tensor(img)
        label = torch.LongTensor(label)
        return img,label

    def __len__(self):
        return len(self.ids)

import cv2
import math
import random
import numpy as np
import Datasets.ModelData as md

class Transform():
    def __call__(self, x, y):
        return self.transform_x(x), self.transform_y(y, x)

    def transform_x(self, x): return x

    def transform_y(self, y, x): return y


class TransformList(Transform):
    '''A class that allows for the composition of transforms
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):       
        for tfm in self.transforms:
            x, y = tfm(x, y) if isinstance(tfm, Transform) else (tfm(x), y)
        return x, y


class RandomTransform(Transform):
    def __call__(self, x, y):
        self.set_state()
        return super().__call__(x, y)

    def set_state(self): raise NotImplementedError


def to_hw(mask):
    cols,rows = np.nonzero(mask)
    if len(cols)==0: return np.zeros(4, dtype=np.float32)
    center_x = (np.max(cols) + np.min(cols)) / 2
    center_y = (np.max(rows) + np.min(rows)) / 2
    width = int(np.max(cols) - np.min(cols))
    height = int(np.max(rows) - np.min(rows))
    return np.array([center_x, center_y, width, height], dtype=np.float32)


class GeometricTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_mask(y, x):
        '''Creates a rectangular mask for the bounding box y on the image x
        
        Arguments:
            y {list} -- [center_x, center_y, width, height]
            x {image tensor} -- Image data
        
        Returns:
            numpy array -- A matrix with width and height that match x
        '''
        r,c,*_ = x.shape
        mask = np.zeros((r, c))
        x_min = int(y[0] - y[2] / 2)
        y_min = int(y[1] - y[3] / 2)
        x_max = int(y[0] + y[2] / 2)
        y_max = int(y[1] + y[3] / 2)

        mask[x_min:x_max, y_min:y_max] = 1.
        return mask

    def transform_y(self, y, x):
        if not isinstance(y, md.StructuredLabel):
            return y

        for i,label in enumerate(y):
            data, data_type = label
            print(data)
            print(data_type)
            if(data_type == md.LabelType.BOUNDING_BOX):
                boxes = [data[i:i+4] for i in range(0, len(data), 4)] # Partition into array of bounding boxes
                masks = [GeometricTransform.make_mask(bb, x) for bb in boxes] # Create masks from bounding boxes
                trfms = [self.transform_x(mask) for mask in masks] # Transform masks
                y[i] = ([to_hw(t) for t in trfms], data_type) # Update Label
        
        return y


class AddPadding(GeometricTransform):
    ''' Adds padding to an image
    '''
    def __init__(self, pad, mode=cv2.BORDER_REFLECT):
        self.pad = pad
        self.mode = mode

    def transform_x(self, im):
        return cv2.copyMakeBorder(im, self.pad, self.pad, self.pad, self.pad, self.mode)


class Scale(GeometricTransform):
    def __init__(self, size):
        self.size = size

    def transform_x(self, im):
        r,c,*_ = im.shape
        mult = self.size/min(r,c)
        sz = (max(math.floor(r*mult), self.size), max(math.floor(c*mult), self.size))
        return cv2.resize(im, sz, interpolation=cv2.INTER_AREA)  


class RandomScale(Scale, RandomTransform):
    def __init__(self, size, scale):
        super().__init__(size)
        self.size = size
        if not isinstance(scale, (list, tuple)): scale = (1, scale)
        self.scale = scale
        self.init_size = size     

    def set_state(self):
        self.size = math.floor(self.init_size * random.uniform(*self.scale))


class CenterCrop(GeometricTransform):
    def __init__(self, size):
        if not isinstance(size, (list, tuple)): size = (size, size)
        self.size = size    


class RandomCrop(RandomTransform, GeometricTransform):
    def __init__(self, size):
        self.size = size

    def set_state(self):
        self.rand_r = random.uniform(0, 1)
        self.rand_c = random.uniform(0, 1)

    def transform_x(self, im):
        r,c,*_ = im.shape
        start_r = np.floor(self.rand_r*(r-self.size)).astype(int)
        start_c = np.floor(self.rand_c*(c-self.size)).astype(int)
        return im[start_r:start_r+self.size, start_c:start_c+self.size]


def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)


class RandomLighting(RandomTransform):
    def __init__(self, b, c):
        self.b,self.c = b,c

    def set_state(self):
        self.b_rand = random.random()*(self.b*2)-self.b
        self.c_rand = random.random()*(self.b*2)-self.c

    def transform_x(self, im):
        b = self.b_rand
        c = self.c_rand
        c = -1/(c-1) if c<0 else c+1
        return lighting(im, b, c)


class RandomHorizontalFlip(RandomTransform, GeometricTransform):
    def set_state(self):
        self.flip = random.random() > .5

    def transform_x(self, im):
        return cv2.flip(im, 1) if self.flip else im


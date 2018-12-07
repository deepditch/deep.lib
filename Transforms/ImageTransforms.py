import cv2
import math
import random
import numpy as np
import Datasets.ModelData as md
import util
import matplotlib.pyplot as plt

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


def center_to_mask(a, im):
    '''Creates a rectangular mask for the bounding box y on an image of width and height
        
    Arguments:
        y {list} -- [center_x, center_y, width, height]
        width {int} -- Width of the image
        height {int} -- Height of the image
    
    Returns:
        numpy array -- A matrix with width and height that match x
    '''
    mask = np.zeros(im.shape[:2],np.uint8)
    bb = center_to_corners(a)
    mask[bb[1]:bb[3], bb[0]:bb[2]] = 1.
    return mask


def mask_to_center(mask):
    '''Converts an image mask to a [center, width, height] bounding box  
    
    Arguments:
        mask {matrix} -- matrix containing an image mask
    
    Returns:
        list -- [center_x, center_y, width, height]
    '''
    points = cv2.findNonZero(mask)
    rect = cv2.boundingRect(points)
    return hw_to_center(rect)


def corners_to_hw(a): 
    """Convert (top left, bottom right) bounding box to (top left, width, height) bounding box.

    Args:
        param1 (arr): [xmin, ymin, xmax, ymax] where (xmin, ymin) 
            and (xmax, ymax) represent the top left and bottom 
            right corners of the bounding box

    Returns:
        arr: [xmin, ymin, width, height]

    """
    bbs = util.partition(a, 4)
    bbs = [[int(a[0]),
            int(a[1]),
            int(a[2]-a[0]),
            int(a[3]-a[1])] for a in bbs]

    return np.concatenate(bbs)

assert (corners_to_hw([2, 5, 10, 20]) == [2, 5, 8, 15]).all()


def corners_to_center(a):
    """Convert (top left, bottom right) bounding box to (center, width, height) bounding box.

    Args:
        param1 (arr): [xmin, ymin, xmax, ymax] where (xmin, ymin) 
            and (xmax, ymax) represent the top left and bottom 
            right corners of the bounding box

    Returns:
        arr: [center_x, center_y, width, height]

    """
    bbs = util.partition(a, 4)
    bbs = [[(a[0]+a[2])/2, 
            (a[1]+a[3])/2, 
            int(a[2]-a[0]), 
            int(a[3]-a[1])] for a in bbs]
    return np.concatenate(bbs)

assert (corners_to_center([2, 5, 10, 20]) == [6, 12.5, 8, 15]).all()


def center_to_hw(a):
    """Convert (center, width, height) bounding box to (top left, width, height) bounding box.

    Args:
        param1 (arr): [center_x, center_y, width, height] where (center_x, center_y) 
            represents the center of the bounding box 

    Returns:
        arr: [xmin, ymin, width, height]

    """
    bbs = util.partition(a, 4)
    bbs = [[int(a[0]-a[2]/2), 
            int(a[1]-a[3]/2), 
            int(a[2]), 
            int(a[3])] for a in bbs]

    return np.concatenate(bbs)

assert (center_to_hw([6, 12.5, 8, 15]) == [2, 5, 8, 15]).all()


def center_to_corners(a):
    """Convert (center, width, height) bounding box to (top left, bottom right) bounding box.

    Args:
        param1 (arr): [center_x, center_y, width, height] where (center_x, center_y) 
            represents the center of the bounding box 

    Returns:
        arr: [xmin, ymin, xmax, ymax]

    """
    bbs = util.partition(a, 4)
    bbs = [[int(a[0]-a[2]/2), 
            int(a[1]-a[3]/2), 
            int(a[0]+a[2]/2), 
            int(a[1]+a[3]/2)] for a in bbs]

    return np.concatenate(bbs)

assert (center_to_corners([6, 12.5, 8, 15]) == [2, 5, 10, 20]).all()

def hw_to_center(a):
    """Convert (top left, width, height) bounding box to (center, width, height) bounding box.

    Args:
        param1 (arr): [xmin, ymin, width, height] where (xmin, ymin) 
            represents the top left of the bounding box 

    Returns:
        arr: [center_x, center_y, width, height]

    """
    bbs = util.partition(a, 4)
    bbs = [[a[0]+a[2]/2, 
            a[1]+a[3]/2, 
            a[2], 
            a[3]] for a in bbs]

    return np.concatenate(bbs)

assert (hw_to_center([2, 5, 8, 15]) == [6, 12.5, 8, 15]).all()

class GeometricTransform(Transform):
    def transform_y(self, y, x):
        if not isinstance(y, md.StructuredLabel):
            return y

        for i, label in enumerate(y):
            data, data_type, name = label

            if(data_type == md.LabelType.CATEGORY):
                y[i] = (np.array(data), data_type, name)

            if(data_type == md.LabelType.BOUNDING_BOX):
                r,c,*_ = x.shape
                boxes = util.partition(data, 4) # Partition into array of bounding boxes
                masks = [center_to_mask(bb, x) for bb in boxes]  # Create masks from bounding boxes
                trfms = [self.transform_x(mask) for mask in masks]  # Transform masks
                y[i] = (np.concatenate([mask_to_center(t) for t in trfms]), data_type, name) # Convert masks back into bounding boxes, save result
        
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


class RandomHorizontalFlip(RandomTransform, GeometricTransform):
    def set_state(self):
        self.flip = random.random() > .5

    def transform_x(self, im):
        return cv2.flip(im, 1) if self.flip else im

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
        self.c_rand = random.random()*(self.c*2)-self.c

    def transform_x(self, x):
        b = self.b_rand
        c = self.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x


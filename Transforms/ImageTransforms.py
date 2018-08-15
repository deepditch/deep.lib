import cv2
import math
import random

class Transform():
    def __call__(self, x, y):
        return self.transform_x(x), self.transform_y(y)

    def transform_x(self, x): return x

    def transform_y(self, y): return y


class TransformList(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):       
        for tfm in self.transforms:
            print(tfm, isinstance(tfm, Transform))
            x, y = tfm(x, y) if isinstance(tfm, Transform) else (tfm(x), y)
        return x, y


class RandomTransform(Transform):
    def __call__(self, x, y):
        self.set_state()
        return super(RandomTransform, self).__call__(x, y)

    def set_state(self): raise NotImplementedError


class CoordTransform(Transform):
    def transform_y(self, y):
        return transform_x(y)


class AddPadding(Transform):
    ''' Adds padding to an image
    '''
    def __init__(self, pad, mode=cv2.BORDER_REFLECT):
        self.pad = pad
        self.mode = mode

    def transform_x(self, im):
        return cv2.copyMakeBorder(im, self.pad, self.pad, self.pad, self.pad, self.mode)


class AddPaddingBB(AddPadding):
    ''' Adds padding to an image and transforms [min x, min y, max x, max y] bounding box label accordingly
    '''
    def transform_y(self, bb):
        bb[0] += self.pad # x_min
        bb[1] += self.pad # y_min
        bb[2] += self.pad # x_max
        bb[3] += self.pad # y_max


class AddPaddingHW(AddPadding):
    ''' Adds padding to an image and transforms [center x, center y, height, width] bounding box label accordingly
    '''
    def transform_y(self, hw):
        bb[0] += self.pad # center_x
        bb[1] += self.pad # center_y


class Scale(Transform):
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
        print(self.size)


class CenterCrop(Transform):
    def __init__(self, size):
        if not isinstance(size, (list, tuple)): size = (size, size)
        self.size = size    


class RandomCrop(RandomTransform):
    def __init__(self, size):
        self.size = size



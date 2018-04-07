from __future__ import print_function, division, absolute_import, unicode_literals
import os
import skimage.data
import numpy as np
from skimage.filters import sobel #, scharr, prewitt,roberts 

class SFSDataProvider(object):
    channels = 3
    def __init__(self):
        # super(SFSDataProvider, self).__init__()
        self._init_data_counter()
        self.images, self.mask, self.normal =self._load_and_format_data()
        self.image_num = self.images.shape[0]
    def _load_and_format_data(self):
        color_dir = '/Users/shensq/Documents/eecs442challenge/train/color/'
        color,_ = self._load_data(color_dir)

        mask_dir = '/Users/shensq/Documents/eecs442challenge/train/mask/'
        mask,_ = self._load_data(mask_dir)

        normal_dir = '/Users/shensq/Documents/eecs442challenge/train/normal'
        normal,_ = self._load_data(normal_dir)

        images = np.zeros((len(color),128,128,3),dtype='f')
        images[...,0] = normalize_d2f(color[...,2])
        images[...,1] = normalize_d2f(mask)
        for i in range(len(color)):
            images[i,:,:,2] = sobel(images[i,:,:,0])
        # mask_ = np.zeros((len(color),128,128,1),dtype='f')
        # mask_[...,0] = mask != 0
        normal = normalize_d2f(normal)
        return images, mask, normal

    def _load_data(self, data_dir):
        data_ = []
        file_order = []
        file_names = [os.path.join(data_dir, f)
            for f in os.listdir(data_dir)]
        file_order =  [ f
            for f in os.listdir(data_dir)]
        for f in file_names:
            data_.append(skimage.data.imread(f))
        data_ = np.array(data_,dtype='f')
        return data_, file_order

    def _init_data_counter(self):
        self.data_counter=0

    def _next_data(self):
        data = self.images[self.data_counter]
        label = self.normal[self.data_counter]
        mask = self.mask[self.data_counter]
        self.data_counter = (self.data_counter+1)%self.image_num
        return data, label, mask

    def __call__(self, n):
        train_data, labels, mask = self._next_data()
        ix,iy,iz = train_data.shape
        ox,oy,oz = labels.shape
        X = np.zeros((n, ix, iy, iz))
        Y = np.zeros((n, ox, oy, oz))
        Z = np.zeros((n, ox, oy, 1))
        X[0] = train_data
        Y[0] = labels
        Z[0] = mask
        for i in range(1, n):
            train_data, labels, mask = self._next_data()
            X[i] = train_data
            Y[i] = labels
            Z[i] = mask
        return X, Y, Z


def normalize_d2f(image):
    return (image/255.0-0.5)*2

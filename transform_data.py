import torch
import numpy as np

from torchvision import transforms
import random
import math
from PIL import Image
#from randaugment import RandAugmentMC

class Numpy2Tensor(object):
    def __call__(self, data):
        return torch.Tensor(data)

class Normalize_spe(object):
    def __init__(self,
                 mean = np.array([3.76501112, 3.75611564]),
                 std = np.array([0.07338769, 0.06449222])):

        self.mean = mean
        self.std = std

    def __call__(self, data):
        for i in range(2):
            data[i] = (data[i] - self.mean[i]) / self.std[i]

        return data

class Normalize_spe_xy(object):
    def __init__(self,
                 max_value = 10.628257178154184,
                 min_value = 0.0011597341927439826):

        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, data):

        data = (data - self.min_value) / (self.max_value - self.min_value)

        return data


class Normalize_img(object):
    def __init__(self, mean=0.29982, std=0.07479776):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img

class Numpy2Tensor_img(object):
    """Convert a 1-channel ``numpy.ndarray`` to 1-c or 3-c tensor,
    depending on the arg parameter of "channels"
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        """
        for SAR images (.npy), shape H * W, we should transform into C * H * W
        :param img:
        :return:
        """
        channels = self.channels
        img_copy = np.zeros([channels, img.shape[0], img.shape[1]])

        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[0], img.shape[1]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.Tensor(img_copy)
            # backward compatibility
            return img_copy.float()

class Numpy2Tensor_spe(object):
    """Convert a 1-channel ``numpy.ndarray`` to 1-c or 3-c tensor,
    depending on the arg parameter of "channels"
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        """
        for SAR images (.npy), shape H * W, we should transform into C * H * W
        :param img:
        :return:
        """
        channels = self.channels

        img_copy = np.zeros([channels, img.shape[1], img.shape[2]])
        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[1], img.shape[2]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.Tensor(img_copy)
            # backward compatibility
            return img_copy.float()


class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, s = 0.2, r1 = 1, mean=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.s = s
        self.r1 = r1

    def __call__(self, img):
        for attempt in range(100):

            area = img.shape[0] * img.shape[1]

            target_area = self.s * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                img[x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class rotate(object):
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, img):
        img = Image.fromarray(img)
        img = img.rotate(self.angle)
        img = np.array(img)
        return img

class noisy(object):
    def __init__(self, var = 0):
        self.var = var

    def __call__(self, img):
        noise = np.random.normal(0, self.var ** 0.5, img.shape)
        img = np.array(img + noise)

        return img


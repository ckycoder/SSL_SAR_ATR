from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.misc
from torchvision import transforms
from randaugment import RandAugmentMC
from PIL import Image

def read_txt(txt_file):
    pd_data = pd.read_csv(txt_file)
    catename2label = pd.read_csv('./MSTAR_128/catename2label_cate10.txt')
    pd_data['label'] = None

    for i in range(len(pd_data)):
        catename = pd_data.loc[i]['catename']
        label = list(catename2label.loc[catename2label['catename'] == catename]['label'])[0]
        pd_data.loc[i]['label'] = label
    return pd_data

class MSTAR(Dataset):
    def __init__(self, txt_file, root_dir, transform = None):
        self.data = read_txt(txt_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = scipy.misc.imread(self.root_dir + self.data.loc[idx]['path'])
        data = data.astype(np.float32)
        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'data': data,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample

class MSTAR_labeled(Dataset):
    def __init__(self, txt_file, img_dir, num_expand, img_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=128,
                                  padding=int(128 * 0.125)),
            ])
        self.index = list(range(len(self.data)))
        self.num_expand = num_expand
        self.labeled_idx = x_u_split(self.index, self.num_expand)

    def __len__(self):
        return self.num_expand
       

    def __getitem__(self, idx):
        slc_img = scipy.misc.imread(self.img_dir + self.data.loc[self.labeled_idx[idx]]['path'])

        catename = self.data.loc[self.labeled_idx[idx]]['catename']
        label = self.data.loc[self.labeled_idx[idx]]['label']
        sample = {'img': slc_img,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[self.labeled_idx[idx]]['path']}
        sample['img'] = Image.fromarray(sample['img'])
        sample['img'] = self.transform_labeled(sample['img'])
        sample['img'] = np.array(sample['img'])
        sample['img'] = sample['img'].astype(np.float32)
        if self.img_transform:
            sample['img'] = self.img_transform(sample['img'])


        return sample

class MSTAR_unlabeled(Dataset):
    def __init__(self, txt_file, img_dir, num_expand, img_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=128,
                                  padding=int(128 * 0.125))])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=128,
                                  padding=int(128 * 0.125)),
            RandAugmentMC(n=2, m=10)
        ])

        self.index = list(range(len(self.data)))
        self.num_expand = num_expand
        self.unlabeled_idx = x_u_split(self.index, self.num_expand)

    def __len__(self):
        return self.num_expand

    def __getitem__(self, idx):
        slc_img = scipy.misc.imread(self.img_dir + self.data.loc[self.unlabeled_idx[idx]]['path'])

        catename = self.data.loc[self.unlabeled_idx[idx]]['catename']
        label = self.data.loc[self.unlabeled_idx[idx]]['label']
        sample = {'weak_img': slc_img,
                  'strong_img':slc_img,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[self.unlabeled_idx[idx]]['path']}
        sample['weak_img'] = Image.fromarray(sample['weak_img'])
        sample['weak_img'] = self.weak(sample['weak_img'])
        sample['weak_img'] = np.array(sample['weak_img'])

        sample['strong_img'] = Image.fromarray(sample['strong_img']).convert('L')
        sample['strong_img'] = self.strong(sample['strong_img'])
        sample['strong_img'] = np.array(sample['strong_img'])


        sample['weak_img'] = sample['weak_img'].astype(np.float32)
        sample['strong_img'] = sample['strong_img'].astype(np.float32)
        if self.img_transform:
            sample['weak_img'] = self.img_transform(sample['weak_img'])
            sample['strong_img'] = self.img_transform(sample['strong_img'])
        return sample

def x_u_split(index, num_expand_x):

    exapand_labeled = num_expand_x // len(index)
    labeled_idx = np.hstack(
        [index for _ in range(exapand_labeled)])

    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    return labeled_idx



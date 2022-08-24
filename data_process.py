import numpy as np
import os
import pandas as pd
import random
from PIL import Image
import scipy.misc

def get_mean_std(spe_root):
    spe_list = os.listdir(spe_root)
    mean = np.array([0, 0])
    std = np.array([0, 0])
    count = 0
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            mean[0] += spectrogram[:,:,16,:].mean()
            mean[1] += spectrogram[:,:,:,16].mean()
            std[0] += spectrogram[:,:,16,:].std()
            std[1] += spectrogram[:,:,:,16].std()
            count += 1

    print(mean / count, std / count)

def get_mean_std_xy(spe_root):
    spe_list = os.listdir(spe_root)
    mean = 0
    std = 0
    count = 0
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            x, y, fr, fa = spectrogram.shape
            spectrogram = spectrogram.reshape([x*y, fr, fa])
            for i in range(x*y):
                mean += spectrogram[i, :, :].mean()
                std += spectrogram[i, :, :].std()
                count += 1

    print(mean / count, std / count)

def get_mean_std_mstar(data_root):
    data_list = os.listdir(data_root)
    mean = 0
    std = 0
    count = 0
    for cate in data_list:
        img_list = os.listdir(data_root + cate)
        for item in img_list:
            img = scipy.misc.imread(data_root + cate + '/'+item)
            print(img.mean())
            mean += img.mean()
            std += img.std()
            count += 1
    print(count)
    print(mean/count,std/count)

def gen_train_val(data_root):
    train_ratio = 0.3#related to 'slc_train(val)_num(1,2,3...22).txt'

    df_train = pd.DataFrame(columns=['path', 'catename'])
    df_val = pd.DataFrame(columns=['path', 'catename'])


    for cate in os.listdir(data_root):
        data_list = os.listdir(data_root + cate)
        random.shuffle(data_list)

        train_num = 10

        for i, item in enumerate(data_list):
            if i < train_num:
                df_train.loc[len(df_train) + 1] = [cate + '/' + item, cate]
            else:
                df_val.loc[len(df_val) + 1] = [cate + '/' + item, cate]

    df_train.to_csv('./MSTAR_128/mstar_train_1.txt', index=False)
    df_val.to_csv('./MSTAR_128/mstar_val_1.txt', index=False)

def gen_test(data_root):

    df_test = pd.DataFrame(columns=['path', 'catename'])
    for cate in os.listdir(data_root):
        data_list = os.listdir(data_root + cate)
        random.shuffle(data_list)

        for i, item in enumerate(data_list):
            df_test.loc[len(df_test) + 1] = [cate + '/' + item, cate]

    df_test.to_csv('./MSTAR_128/mstar_test.txt', index=False)

def gen_mstar(data_root):
    label = -1
    df_test = pd.DataFrame(columns=['path', 'label'])
    for cate in os.listdir(data_root):
        label = label + 1
        data_list = os.listdir(data_root + cate + '/'+cate+'JPG')
        random.shuffle(data_list)
        for i, item in enumerate(data_list):
            df_test.loc[len(df_test) + 1] = [test_root + cate + '/'+cate+'JPG' + '/' + item, label]
    print('1')
    df_test.to_csv('../MSTAR/mstar_test.txt', index=False)

if __name__ == '__main__':
    #slc_root = '../data/slc_data_hh/' no this root
    #slc_root = '../data/slc_data/'
    #spe_root = '../data/sub_looks_data/'
    #gen_all_sublooks(slc_root, spe_root)#
    train_root = './MSTAR_128/17DEG/'
    test_root = './MSTAR_128/15DEG/'

    get_mean_std_mstar(train_root)

import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from torch import nn
import argparse

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        if label[i] == 7 or label[i] == 17:
        # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1((label[i]+1) / 20),
                            fontdict={'weight': 'bold', 'size': 7})
        #plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(1 / 10),
                    #fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--training_dataset', default='non_uniform')
    parser.add_argument('--range', default='1')
    args = parser.parse_args()
    datasetnum = args.training_dataset
    range_ratio = args.range
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = {'train': './MSTAR_128/' + datasetnum + '/mstar_train_' + range_ratio + '.txt',
                'val': './MSTAR_128/' + datasetnum + '/mstar_val_' + range_ratio + '.txt',
                'test': './MSTAR_128/mstar_test.txt'}

    batch_size = {'labeled': 100,
                  'unlabeled': 100,
                  'test': 50,
                  'un': 50}
    cate_num = 10
    save_model_path = './MSTAR_128/' + datasetnum + '/' + range_ratio + '/model_ssl/slc_joint_deeper_img_' + str(
        datasetnum) + '_FR_'

    img_transform = transforms.Compose([
        transform_data.Normalize_img(mean=7.4924281121034, std=9.0668273625587),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = {'labeled': slc_dataset.MSTAR_labeled(txt_file=txt_file['train'],
                                                    img_dir='./MSTAR_128/17DEG/',
                                                    num_expand=2647,
                                                    img_transform=img_transform,
                                                    ),

               'unlabeled': slc_dataset.MSTAR_unlabeled(txt_file=txt_file['val'],
                                                        img_dir='./MSTAR_128/17DEG/',
                                                        num_expand=2647,
                                                        img_transform=img_transform),

               'test': slc_dataset.MSTAR(txt_file=txt_file['test'],
                                         root_dir='./MSTAR_128/15DEG/',
                                         transform=img_transform, ),
               'un': slc_dataset.MSTAR(txt_file=txt_file['val'],
                                       root_dir='./MSTAR_128/17DEG/',
                                       transform=img_transform, )
               }

    train_count_dict = {}
    for i in range(cate_num):
        train_count_dict[i] = len(dataset['labeled'].data.loc[dataset['labeled'].data['label'] == i])
        # print(train_count_dict){0: 330, 1: 176, 2: 285, 3: 183, 4: 312, 5: 332, 6: 302, 7: 310}

    loss_weight = [
        (1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
        for i in range(cate_num)]
    # print(loss_weight)#[0.9737347853939783, 1.0526585522101217, 0.9967969250480461, 1.0490711082639332, 0.9829596412556053, 0.972709801409353, 0.9880845611787316, 0.9839846252402307]
    dataloaders = {}
    dataloaders['labeled'] = DataLoader(dataset['labeled'],
                                        batch_size=batch_size['labeled'],
                                        sampler=ImbalancedDatasetSampler(dataset['labeled'],
                                                                         indices=dataset['labeled'].labeled_idx),
                                        num_workers=0)

    dataloaders['unlabeled'] = DataLoader(dataset['unlabeled'],
                                          batch_size=batch_size['unlabeled'],
                                          shuffle=True,
                                          num_workers=0)
    dataloaders['test'] = DataLoader(dataset['test'],
                                     batch_size=batch_size['test'],
                                     num_workers=0)
    dataloaders['un'] = DataLoader(dataset['un'],
                                   batch_size=batch_size['un'],
                                   num_workers=0)
    net_joint = network.ResNet18_ssl(cate_num)
    # net_joint_dict = net_joint.state_dict()
    # pretrained_dict = torch.load('./model_1_arc/slc_joint_deeper_img__1_FR_epoch8000.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_joint_dict}
    # net_joint_dict.update(pretrained_dict)
    # net_joint.load_state_dict(net_joint_dict)
    net_joint.load_state_dict(torch.load('/data/cky/semi_v2/MSTAR_128/non_uniform/1/model_ssl/slc_joint_deeper_img_non_uniform_FR_epoch4000.pth'))
    net_joint.to(device)


    acc_num = 0.0
    data_num = 0
    val_loss = 0.0

    net_joint.eval()
    iter_val = iter(dataloaders['test'])
    y_score = np.zeros([0, 10])
    y_label = np.zeros([0], dtype=int)
    print('2222')

    net_joint.eval()
    iter_val = iter(dataloaders['labeled'])
    y_score = np.zeros([0,512])
    y_label = np.zeros([0], dtype=int)
    for j in range(len(dataloaders['labeled'])):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_label = val_data['label'].to(device)

        val_output = net_joint(val_img,64)

        #val_output = net_joint(val_img,val_img.shape[0])

        #y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)
        y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)

        y_score = np.concatenate((y_score, val_output.cpu().data.numpy()))

        _, pred = torch.Tensor.max(val_output, 1)
        acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
        data_num += val_label.size()[0]
        #val_loss += loss_func(val_output, val_label).item()

    val_acc = acc_num / data_num
    #print(y_score.shape)
    print(val_acc)
    net_joint.eval()
    iter_val = iter(dataloaders['unlabeled'])
    print(len(dataloaders['unlabeled']))
    for j in range(len(dataloaders['unlabeled'])):
        val_data = next(iter_val)
        val_img = val_data['weak_img'].to(device)
        val_label = val_data['label'].to(device)

        val_output = net_joint(val_img,64)

        #val_output = net_joint(val_img,val_img.shape[0])

        #y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)
        y_label = np.concatenate((y_label, val_label.cpu().data.numpy()+10), axis=0)

        y_score = np.concatenate((y_score, val_output.cpu().data.numpy()))

        _, pred = torch.Tensor.max(val_output, 1)
        acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
        data_num += val_label.size()[0]
    print('1111')
    print(y_label)
    ts = TSNE(n_components=2, init='pca', random_state=0)
    reslut = ts.fit_transform(y_score)
    fig = plot_embedding(reslut, y_label, 't-SNE Embedding of digits')
    plt.savefig('./fm_bfda.png')
    plt.show()

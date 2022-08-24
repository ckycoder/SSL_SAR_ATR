import network
import transform_data
import slc_dataset as slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
from torch import nn
from torch import optim
import argparse
from losses.svm import SmoothSVM
import os
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train')
    parser.add_argument('--training_dataset', default='uniform')
    parser.add_argument('--range', default='5')
    args = parser.parse_args()
    datasetnum = args.training_dataset
    range_ratio = args.range

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = {'train': './MSTAR_128/' + datasetnum + '/mstar_train_' + range_ratio + '.txt',
                'val': './MSTAR_128/' + datasetnum + '/mstar_val_' + range_ratio + '.txt',
                'test': './MSTAR_128/mstar_test.txt'}

    batch_size = {'labeled': 100,
                  'unlabeled': 1,
                  'test':50}
    cate_num = 10
    save_model_path = './MSTAR_128/' + datasetnum +'/' + range_ratio +'/model_res/slc_joint_deeper_img_' + str(datasetnum) + '_FR_'


    img_transform = transforms.Compose([
        transform_data.Normalize_img(mean=7.4924281121034,std= 9.0668273625587),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = {'labeled' : slc_dataset.MSTAR_labeled(txt_file=txt_file['train'],
                                       img_dir='./MSTAR_128/17DEG/',
                                        num_expand=2747,
                                       img_transform=img_transform,
                                    ),

               'unlabeled' : slc_dataset.MSTAR(txt_file=txt_file['val'],
                                            root_dir='./MSTAR_128/17DEG/',
                                            transform=img_transform, ),

               'test': slc_dataset.MSTAR(txt_file=txt_file['test'],
                                        root_dir='./MSTAR_128/15DEG/',
                                         transform=img_transform,)

               }

    train_count_dict = {}
    for i in range(cate_num):
        train_count_dict[i] = len(dataset['labeled'].data.loc[dataset['labeled'].data['label'] == i])


    loss_weight = [(1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
                       for i in range(cate_num)]

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset['labeled'],
                                      batch_size=batch_size['labeled'],
                                      sampler=ImbalancedDatasetSampler(dataset['labeled'],indices = dataset['labeled'].labeled_idx),
                                      num_workers=0)

    dataloaders['unlabeled'] = DataLoader(dataset['unlabeled'],
                                    batch_size=batch_size['unlabeled'],
                                    num_workers=0)
    dataloaders['test'] = DataLoader(dataset['test'],
                                    batch_size=batch_size['test'],
                                    num_workers=0)

    net_joint = network.ResNet18_TSX(10)
    net_joint.to(device)

    epoch_num = 4000
    i = 0

    optimizer = optim.Adam(net_joint.parameters(), lr=0.0001, weight_decay=0.004)


    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    print(lr_list)
    loss_weight = torch.Tensor(loss_weight).to(device)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)

    result_dir = './MSTAR_128/' + datasetnum +'/' + range_ratio
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    f_acc = open(result_dir+'/accuracy.txt','w')

    for epoch in range(epoch_num):
        ratio = 0
        for data in dataloaders['train']:
            net_joint.train()
            optimizer.zero_grad()

            img_data = data['img'].to(device)
            labels = data['label'].to(device)


            output = net_joint(img_data)
            loss = loss_func(output, labels)

            loss.backward()
            optimizer.step()
            i += 1



        print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', loss.item())
        net_joint.eval()
        acc_num = 0.0
        data_num = 0
        val_loss = 0.0
        iter_val = iter(dataloaders['unlabeled'])


        acc_num = 0.0
        data_num = 0
        val_loss = 0.0
        iter_val = iter(dataloaders['test'])
        for j in range(len(dataloaders['test'])):
            val_data = next(iter_val)
            val_img = val_data['data'].to(device)
            val_label = val_data['label'].to(device)

            #_, val_output = net_joint(val_img,val_label)
            val_output = net_joint(val_img)

            # val_loss = loss_func(val_output, val_label)
            _, pred = torch.Tensor.max(val_output, 1)
            acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
            data_num += val_label.size()[0]
            val_loss += loss_func(val_output, val_label).item()

        val_loss /= len(dataloaders['test'])
        val_acc = acc_num / data_num
        print(val_acc)
        f_acc.write(str(val_acc.item()) + '\n')

    f_acc.close()

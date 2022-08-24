import network
import transform_data
import slc_dataset as slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch import optim
import argparse
from losses.svm import SmoothSVM
import os

LABELED_FM_TABELS = None
UNLABELED_FM_TABELS = None
#loss_function
def get_kernel(x,y):
    x_num = x.size(0)
    y_num = y.size(0)
    assert x.size(1) == y.size(1)
    fm_dim = x.size(1)
    x = x.unsqueeze(1)  # (x_num, 1, fm_dim)
    y = y.unsqueeze(0)  # (1, y_num, fm_dim)
    #为了保持x,y的数目size相等
    x_expand = x.expand(x_num,y_num,fm_dim)
    y_expand = y.expand(x_num,y_num,fm_dim)

    kernel = torch.exp(-(x_expand-y_expand).pow(2).mean(2))
    return kernel.mean()

def mmd(x,y):
    x_kernel = get_kernel(x,x)
    y_kernel = get_kernel(y,y)
    xy_kernel = get_kernel(x,y)
    mmd_loss = x_kernel + y_kernel -2*xy_kernel
    return mmd_loss

def get_fm_tabel(fm_l,fm_u,mask_l,mask_u,table_size):
    global LABELED_FM_TABELS
    global UNLABELED_FM_TABELS
    if mask_l is not None:

        mask_l = mask_l.nonzero().squeeze(1)
        mask_u = mask_u.nonzero().squeeze(1)

        fm_l = fm_l[mask_l]
        fm_u = fm_u[mask_u]


    if table_size > 0:

        if LABELED_FM_TABELS is None:
            LABELED_FM_TABELS = fm_l
            UNLABELED_FM_TABELS = fm_u
        else:
            LABELED_FM_TABELS = torch.cat([LABELED_FM_TABELS, fm_l])
            UNLABELED_FM_TABELS = torch.cat([UNLABELED_FM_TABELS, fm_u])
            if len(LABELED_FM_TABELS) > table_size:
                LABELED_FM_TABELS = LABELED_FM_TABELS[-table_size:]
            if len(UNLABELED_FM_TABELS) > table_size:
                UNLABELED_FM_TABELS = UNLABELED_FM_TABELS[-table_size:]

        fm_l = LABELED_FM_TABELS
        fm_u = UNLABELED_FM_TABELS
        LABELED_FM_TABELS = LABELED_FM_TABELS.detach()
        UNLABELED_FM_TABELS = UNLABELED_FM_TABELS.detach()

    return fm_l,fm_u


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--training_dataset', default='non_uniform')
    parser.add_argument('--range', default='1')
    parser.add_argument('--topk', default=3)
    args = parser.parse_args()
    datasetnum = args.training_dataset
    range_ratio = args.range
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = {'train': './MSTAR_128/' + datasetnum + '/mstar_train_' + range_ratio + '.txt',
                'val': './MSTAR_128/' + datasetnum + '/mstar_val_' + range_ratio + '.txt',
                'test': './MSTAR_128/mstar_test.txt'}

    batch_size = {'labeled': 10,
                  'unlabeled':10,
                  'test':50,
                  'un':50}
    cate_num = 10
    save_model_path = './MSTAR_128/' + datasetnum +'/' + range_ratio +'/model_top3_mmd/slc_joint_deeper_img_' + str(datasetnum) + '_FR_'


    img_transform = transforms.Compose([
        transform_data.Normalize_img(mean=7.4924281121034,std= 9.0668273625587),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = {'labeled': slc_dataset.MSTAR_labeled(txt_file=txt_file['train'],
                                                    img_dir='./MSTAR_128/17DEG/',
                                                    num_expand=2647,
                                                    img_transform=img_transform,
                                                    ),

               'unlabeled': slc_dataset.MSTAR_unlabeled(txt_file=txt_file['val'],
                                              img_dir='./MSTAR_128/17DEG/',
                                              num_expand=2647*7,
                                              img_transform=img_transform),

               'test': slc_dataset.MSTAR(txt_file=txt_file['test'],
                                         root_dir='./MSTAR_128/15DEG/',
                                         transform=img_transform, ),
               'un': slc_dataset.MSTAR(txt_file=txt_file['val'],
                                       root_dir='./MSTAR_128/17DEG/',
                                       transform=img_transform,)
               }

    train_count_dict = {}
    for i in range(cate_num):
        train_count_dict[i] = len(dataset['labeled'].data.loc[dataset['labeled'].data['label'] == i])
        #print(train_count_dict){0: 330, 1: 176, 2: 285, 3: 183, 4: 312, 5: 332, 6: 302, 7: 310}

    loss_weight = [(1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
                       for i in range(cate_num)]
    #print(loss_weight)#[0.9737347853939783, 1.0526585522101217, 0.9967969250480461, 1.0490711082639332, 0.9829596412556053, 0.972709801409353, 0.9880845611787316, 0.9839846252402307]
    dataloaders = {}
    dataloaders['labeled'] = DataLoader(dataset['labeled'],
                                      batch_size=batch_size['labeled'],
                                      sampler=ImbalancedDatasetSampler(dataset['labeled'],indices = dataset['labeled'].labeled_idx),
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
    net_joint.to(device)

    epoch_num = 4000
    i = 0
    optimizer = optim.Adam(net_joint.parameters(), lr=0.0001, weight_decay=0.004)
    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    print(lr_list)
    loss_weight = torch.Tensor(loss_weight).to(device)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)
    topk_criterion = SmoothSVM(n_classes=10, k=int(args.topk), tau=1, alpha=1,loss_weight=loss_weight)

    result_dir = './MSTAR_128/' + datasetnum + '/' + range_ratio
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    f_acc = open(result_dir + '/top3_mmd_accuracy_pretrain.txt','w')
    #u_acc = open(result_dir + '/u_top3_mmd_accuracy.txt', 'w')


    for epoch in range(epoch_num):
        ratio = 0.2
        train_loader = zip(dataloaders['labeled'], dataloaders['unlabeled'])
        net_joint.train()
        for (data_labeled, data_unlabeled) in train_loader:

            optimizer.zero_grad()

            labeled_img = data_labeled['img']
            weak_img = data_unlabeled['weak_img']
            strong_img = data_unlabeled['strong_img']

            unlabeled_labels = data_unlabeled['label'].to(device)
            labels = data_labeled['label'].to(device)
            all_img = torch.cat((labeled_img, weak_img, strong_img), 0).to(device)

            sup_fm, unsup_fm,sup_logits, unsup_logits = net_joint(all_img, labels.shape[0], is_training=True)

            weak_logits = unsup_logits[:weak_img.shape[0]]
            strong_logits = unsup_logits[weak_img.shape[0]:]
            weak_fm = unsup_fm[:weak_img.shape[0]]
            strong_fm = unsup_fm[:weak_img.shape[0]]
            del unsup_logits
            pseudo_label_u = F.softmax(weak_logits.detach(), dim=-1)
            max_probs_u, targets_u = torch.max(pseudo_label_u, dim=-1)
            pseudo_label_l = F.softmax(sup_logits.detach(), dim=-1)
            max_probs_l, targets_l = torch.max(pseudo_label_l, dim=-1)
            threshod = 0.8

            mask_u = max_probs_u.ge(threshod).float()
            mask_l = max_probs_l.ge(threshod).float()


            unsup_ce_loss = (F.cross_entropy(strong_logits, targets_u, reduce=False) * mask_u).mean()
            unsup_topk_loss = (topk_criterion(strong_logits,targets_u)* mask_u).mean()
            unsup_loss = ((1.0 - ratio) * unsup_ce_loss + ratio * unsup_topk_loss)


            table_size = 64
            sup_loss = loss_func(sup_logits, labels)

            mmd_loss = torch.zeros_like(sup_loss)
            if mask_l.sum() > 0 and mask_u.sum() > 0:
                fm_l, fm_u = get_fm_tabel(sup_fm, weak_fm, mask_l, mask_u, table_size)
                if len(fm_l) > 20:
                    mmd_loss = mmd(sup_fm, weak_fm)

            total_loss= unsup_loss + sup_loss+ mmd_loss #+ kl_loss
            total_loss.backward()
            optimizer.step()
            i += 1

        acc_num = 0.0
        data_num = 0
        val_loss = 0.0

        print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', total_loss.item(),mmd_loss.item())


        acc_num = 0.0
        data_num = 0
        val_loss = 0.0
        iter_val = iter(dataloaders['test'])
        for j in range(len(dataloaders['test'])):
            val_data = next(iter_val)
            val_img = val_data['data'].to(device)
            val_label = val_data['label'].to(device)
            val_output = net_joint(val_img, val_img.shape[0])
            _, pred = torch.Tensor.max(val_output, 1)
            acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
            data_num += val_label.size()[0]
            val_loss += loss_func(val_output, val_label).item()

        val_loss /= len(dataloaders['test'])
        val_acc = acc_num / data_num
        print(val_acc)
        f_acc.write(str(val_acc.item()) + '\n')

    f_acc.close()
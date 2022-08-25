# SSL_SAR_ATR
        The code of the paper ‘Learning from Reliable Unlabeled Samples for Semi-supervised SAR ATR’.
        published on 'IEEE Geoscience and Remote Sensing Letters'.
# Abstract
Synthetic aperture radar automatic target recognition (SAR ATR) has been suffering from the insufficient labeled samples as the annotation of SAR data is time-consuming. Thus, adding unlabeled samples into training has attracted the attention of researchers. In this letter, a semi-supervised method based on consistency criterion, domain adaptation (DA), and Top- k loss is proposed to alleviate the need for labeled samples. According to consistency criterion that samples generated by the weak and strong augmentations (WSAs) from the same sample belong to the same category, we use the weak and strong augmented unlabeled samples to predict pseudo labels and train the model, respectively. Then, to overcome the issue caused by the domain discrepancy between labeled and unlabeled samples especially when labeled samples concentrate on a narrow azimuth range, a DA component is designed to reduce their discrepancy. Besides, considering the incorrect pseudo labels will hamper the model training, the Top- k loss is adopted for unlabeled samples to mitigate the negative effects. The experimental results on moving and stationary target acquisition and recognition (MSTAR) dataset demonstrate the superiority of our method in semi-supervised SAR ATR. Specifically, we achieve about a 14.29% improvement in recognition accuracy compared to the state-of-the-art when the labeled samples concentrate on a narrow azimuth range.
        
# dataset
        
# Run 
        run train_XXX.py will give the recognition accuracy on the test set in every epoch （pyhton train_XXX.py）
        run test_ResNet.py will give the t-SNE based visualization of feature maps. （pyhton test_ResNet.py）
        
## parameters
        --training_dataset represents the sampling strategy：uniform or non_uniform
        --range represents the selective of labeled samples:
        1.uniform 5,10,20   --> TABLE I of the paper
        2.uniform 0,1,2,3,...,11 ---- 0-30, 30-60, 60-90,....,330-360,  --> TABLE II of the paper
## Different components
        train_ResNet.py -- Basic_Network
        train_ResNet_WSA.py -- WSAs
        train_ResNet_WSA_mmd.py -- WSAs+DA
        train_ResNet_WSA_topk.py -- WSA+Top-k
        train_ResNet_WSA_topk_mmd.py -- WSA+DA+Top-k
        
        
# Citation

### If you find this repository/work helpful in your research, welcome to cite the paper.
        @ARTICLE{9853536,
        author={Chen, Keyang and Pan, Zongxu and Huang, Zhongling and Hu, Yuxin and Ding, Chibiao},
        journal={IEEE Geoscience and Remote Sensing Letters}, 
        title={Learning From Reliable Unlabeled Samples for Semi-Supervised SAR ATR}, 
        year={2022},
        volume={19},
        number={},
        pages={1-5},
        doi={10.1109/LGRS.2022.3197892}}

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from base_dvc import BaseDVC
import os
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import random_split


class FMNISTDVC(BaseDVC):
    def __init__(self, entropy_weight, con_entropy_weight,seed):
        super(FMNISTDVC, self).__init__(
            input_dim=28*28,  # FMNIST输入维度
            latent_dim=10,   # 潜在空间维度
            num_clusters=10,  # FMNIST类别数
            dataset_name="fmnist",  # 数据集名称
            entropy_weight=entropy_weight,  # 熵损失权重
            con_entropy_weight=con_entropy_weight,  # 条件熵损失权重
            seed = seed
        )
        self._set_seed()
        # 设置训练参数
        self.pretrain_epochs = 100  # 预训练轮数
        self.train_epochs = 100     # 训练轮数
        self.pretrain_lr = 1e-4     # 预训练学习率
        self.train_lr = 1e-3        # 训练学习率
        
        
        # 编码器
        self.conv1 = nn.Conv2d(1, 32, 5)  # 14, 14, 32
        self.pool = nn.AvgPool2d(2, 2)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)  # 7, 7, 64
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)  # 3, 3, 128
        self.batch3 = nn.BatchNorm2d(128)
        
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(2*2*128, 10)  # mu
        self.linear2 = nn.Linear(2*2*128, 10)  # logvar
        
        # 解码器
        self.linear1_de = nn.Linear(10, 2*2*128)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 2, 2))
        
        self.convT1 = nn.ConvTranspose2d(128, 64, 3)
        self.upsample1 = nn.UpsamplingBilinear2d(size=(8, 8))
        self.batch5 = nn.BatchNorm2d(64)
        self.convT2 = nn.ConvTranspose2d(64, 32, 5)
        self.upsample2 = nn.UpsamplingBilinear2d(size=(24, 24))
        self.batch6 = nn.BatchNorm2d(32)
        self.convT3 = nn.ConvTranspose2d(32, 1, 5)

        self.pred = nn.Linear(10, 10)
    
    def encoder(self, x):
        # 确保输入是4D张量 [batch_size, channels, height, width]
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)
            
        l1 = self.pool(F.relu(self.conv1(x)))
        l1 = self.batch1(l1)
        l2 = self.pool(F.relu(self.conv2(l1)))
        l2 = self.batch2(l2)
        l3 = F.relu(self.conv3(l2))
        l3 = self.batch3(l3)
        l3 = self.flatten(l3)
        
        mu = self.linear1(l3)
        logvar = self.linear2(l3)
        return mu, logvar
    
    def decoder(self, x):
        embedding = self.linear1_de(x)
        l4 = self.unflatten(embedding)
        l5 = self.upsample1(F.relu(self.convT1(l4)))
        l5 = self.batch5(l5)
        l6 = self.upsample2(F.relu(self.convT2(l5)))
        l6 = self.batch6(l6)
        l7 = torch.sigmoid(self.convT3(l6))
        return l7
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        pred = F.softmax(self.pred(mu), dim=1)
        return mu, logvar, pred, z
    
    
    def _build_encoder(self):
        """这个方法不再使用，但需要实现以满足父类要求"""
        return None, None, None
    
    def _build_decoder(self):
        """这个方法不再使用，但需要实现以满足父类要求"""
        return None
    
    def loss_function(self, mu, logvar, pred,recon_x,x):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        pred_weight = self.state_dict()['pred.weight']
        element_1 = mu.unsqueeze(1)
        element_2 = pred_weight.unsqueeze(0)
        mu_v = element_1 - element_2
        logvar_v = logvar.unsqueeze(1)
        pred_v = pred.unsqueeze(2)
        KLD_element = mu_v.pow(2).mul(pred_v) + logvar_v.exp() -1 - logvar_v
        KLD = torch.sum(KLD_element).mul_(0.5)
        f_j = torch.sum(pred, axis=0)
        entropy = torch.sum(torch.mul(f_j, torch.log(f_j)))
        con_entropy = -1 * torch.sum(torch.mul(pred, torch.log(pred + 1.0e-8)))

        return KLD,con_entropy,entropy, KLD + self.con_entropy_weight * con_entropy + self.entropy_weight * entropy
    
    @staticmethod
    def get_data_loaders(batch_size=128, data_dir='Datasets'):
        """获取FMNIST数据加载器"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307,], std=[0.3081,])
        ])
        
        # 创建数据集
        train_dataset = torchvision.datasets.FashionMNIST(
            data_dir, transform=transform, train=True, download=False
        )
        test_dataset_all = torchvision.datasets.FashionMNIST(
            data_dir, transform=transform, train=False, download=False
        )
        total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset_all])
        test_dataset, _ = random_split(test_dataset_all, [10000, 0])
        
        # 创建数据加载器
        # 配置数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # 使用多进程加载
            pin_memory=True,  # 使用固定内存
            persistent_workers=True,  # 保持工作进程存活
            prefetch_factor=2,  # 预取因子
            # drop_last=True  # 丢弃不完整的批次
        )
        
        test_loader = DataLoader(
            test_dataset,
            #total_dataset,
            batch_size=10000,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            # drop_last=False  # 测试集不需要丢弃
        )
        
        return train_loader, test_loader 
    
    
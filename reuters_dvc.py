import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from base_dvc import BaseDVC
import os
import numpy as np

class ReutersDataset(Dataset):
    def __init__(self, data_dir='Datasets'):
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(root_dir, data_dir)
        self.data = np.load(os.path.join(self.data_dir, 'reutersidf10k.npy'), allow_pickle=True).item()
    
    def __getitem__(self, index):
        img = torch.tensor(self.data['data'][index]).to(torch.float32)
        lab = self.data['label'].reshape(-1)[index]
        return img, lab
    
    def __len__(self):
        return len(self.data['data'])

class ReutersDVC(BaseDVC):
    def __init__(self, entropy_weight, con_entropy_weight,seed):
    #def __init__(self, entropy_weight=1, con_entropy_weight=10,seed = 192):
        super(ReutersDVC, self).__init__(
            input_dim=2000,  # Reuters输入维度
            latent_dim=4,   # 潜在空间维度
            num_clusters=4,  # Reuters类别数
            dataset_name="reuters",  # 数据集名称
            entropy_weight=entropy_weight,  # 熵损失权重
            con_entropy_weight=con_entropy_weight,  # 条件熵损失权重
            seed = seed
        )
        self._set_seed()
        # 设置训练参数
        self.pretrain_epochs = 100  # 预训练轮数
        self.train_epochs = 200     # 训练轮数
        self.pretrain_lr = 1e-4     # 预训练学习率
        self.train_lr = 1e-5        # 训练学习率
        
        
        # 编码器
        self.fc1 = nn.Linear(2000, 500)
        self.bn1 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc3 = nn.Linear(500, 2000)
        self.bn3 = nn.BatchNorm1d(num_features=2000, affine=False)
        self.fc4_mu = nn.Linear(2000, 4)
        self.bn4_mu = nn.BatchNorm1d(num_features=4)
        self.fc4_logvar = nn.Linear(2000, 4)
        self.bn4_logvar = nn.BatchNorm1d(num_features=4)

        # 解码器
        self.fc5 = nn.Linear(4, 2000)
        self.bn5 = nn.BatchNorm1d(num_features=2000, affine=False)
        self.fc6 = nn.Linear(2000, 500)
        self.bn6 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc7 = nn.Linear(500, 500)
        self.bn7 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc8 = nn.Linear(500, 2000)
        self.bn8 = nn.BatchNorm1d(num_features=2000)
        
        self.pred = nn.Linear(4, 4)
    
    def encoder(self, x):
        l1 = F.relu(self.bn1(self.fc1(x)))
        l2 = F.relu(self.bn2(self.fc2(l1)))
        l3 = F.relu(self.bn3(self.fc3(l2)))
        mu = self.fc4_mu(l3)
        logvar = self.fc4_logvar(l3)
        return mu, logvar
    
    def decoder(self, x):
        l4 = F.relu(self.bn5(self.fc5(x)))
        l5 = F.relu(self.bn6(self.fc6(l4)))
        l6 = F.relu(self.bn7(self.fc7(l5)))
        l7 = self.fc8(l6)
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
    
    
    @staticmethod
    def get_data_loaders(batch_size=128, data_dir='Datasets'):
        """获取Reuters数据加载器"""
        # 创建数据集
        dataset = ReutersDataset(data_dir)
        
        # 创建数据加载器
        # 配置数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=1,  # 使用多进程加载
            # pin_memory=True,  # 使用固定内存
            # persistent_workers=True  # 保持工作进程存活
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=11228,
            shuffle=False,
            # num_workers=1,
            # pin_memory=True,
            # persistent_workers=True
        )
        
        return train_loader, test_loader 
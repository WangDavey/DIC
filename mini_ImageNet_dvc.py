import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from base_dvc import BaseDVC
import os
from torch.utils.data.dataset import random_split

class MiniImageNetDataset(Dataset):
    def __init__(self, data_dir='Datasets'):
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(root_dir, data_dir)
        self.data = loadmat(os.path.join(self.data_dir, 'mini_imagenet_features.mat'))
    
    def __getitem__(self, index):
        img = torch.tensor(self.data['fea'][index]).to(torch.float32)
        lab = self.data['true'].reshape(-1)[index]
        return img, lab
    
    def __len__(self):
        return len(self.data['fea'])

class MiniImageNetDVC(BaseDVC):
    def __init__(self, entropy_weight, con_entropy_weight, seed):
        super(MiniImageNetDVC, self).__init__(
            input_dim=2048,  # ResNet50最后一层特征维度
            latent_dim=100,   # 潜在空间维度
            num_clusters=100,  # Mini-ImageNet类别数
            dataset_name="mini_imageNet",  # 数据集名称
            entropy_weight=entropy_weight,  # 熵损失权重
            con_entropy_weight=con_entropy_weight,  # 条件熵损失权重
            seed=seed
        )
        self._set_seed()
        
        # 设置训练参数
        self.pretrain_epochs = 100  # 预训练轮数
        self.train_epochs = 100     # 训练轮数
        self.pretrain_lr = 1e-4     # 预训练学习率
        self.train_lr = 1e-5        # 训练学习率
        
        # 覆盖父类的层定义，使用与原始模型相同的结构
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024, affine=False)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512, affine=False)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc4_mu = nn.Linear(256, 100)
        self.bn4_mu = nn.BatchNorm1d(num_features=100)
        self.fc4_logvar = nn.Linear(256, 100)
        self.bn4_logvar = nn.BatchNorm1d(num_features=100)

        self.fc5 = nn.Linear(100, 256)
        self.bn5 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc6 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(num_features=512, affine=False)
        self.fc7 = nn.Linear(512, 1024)
        self.bn7 = nn.BatchNorm1d(num_features=1024, affine=False)
        self.fc8 = nn.Linear(1024, 2048)
        self.bn8 = nn.BatchNorm1d(num_features=2048)
        
        self.pred = nn.Linear(100, 100)
    
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
    def get_data_loaders(batch_size=512, data_dir='Datasets'):
        """获取Mini-ImageNet数据加载器"""
        # 创建数据集
        train_dataset = MiniImageNetDataset(data_dir)
        test_dataset_all = MiniImageNetDataset(data_dir)
        
        # 获取数据集长度
        total_length = len(test_dataset_all)
        test_length = min(10000, total_length)  # 确保不超过数据集长度
        remaining_length = total_length - test_length
        
        # 分割数据集
        test_dataset, _ = random_split(test_dataset_all, [test_length, remaining_length])
        
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
            #test_dataset_all,
            batch_size=10000,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            # drop_last=False  # 测试集不需要丢弃
        )
        
        return train_loader, test_loader 
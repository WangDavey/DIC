import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from base_dvc import BaseDVC
import os
import numpy as np
import scanpy as sc
import h5py
from typing import Optional
from utils.preprocess import prepro, normalize_1


GENESET_CONFIGS = {
    'Quake_Smart-seq2_Limb_Muscle': (6,1090),
    'Quake_Smart-seq2_Trachea': (4,1350),
    'Quake_Smart-seq2_Diaphragm': (5,870),
    'Adam':(8,3660),
    'Quake_10x_Limb_Muscle': (6,3909),
    'Romanov': (7,2881),
}

class GenomeDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, n_input: int = 2000):
        """
        初始化基因数据集
        Args:
            data_dir: 数据目录
            dataset_name: 数据集名称
            n_input: 高变基因数量
            preorder: 是否进行预排序
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.n_input = n_input
        
        # 加载和预处理数据
        x, y = prepro(os.path.join(data_dir, f'{dataset_name}/data.h5'))
        x = np.ceil(x).astype(int)
        adata = sc.AnnData(x)
        adata.obs['Group'] = y
        adata = normalize_1(adata, copy=True, highly_genes=n_input)
        

        count = adata.X
            
        self.x = torch.FloatTensor(count)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class GenomeDVC(BaseDVC):
    def __init__(self, 
                 geneset_name,
                 input_dim,
                 entropy_weight,
                 con_entropy_weight,
                 seed):
        """
        初始化基因数据DVC模型
        Args:
            geneset_name: 基因数据集名称
            input_dim: 输入维度（高变基因数量）
            entropy_weight: 熵损失权重
            con_entropy_weight: 条件熵损失权重
            seed: 随机种子
        """
        # 根据geneset_name获取配置
        if geneset_name not in GENESET_CONFIGS:
            raise ValueError(f"不支持的基因数据集: {geneset_name}")
        num_clusters, _ = GENESET_CONFIGS[geneset_name]
        latent_dim = num_clusters
        super(GenomeDVC, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_clusters=num_clusters,
            dataset_name=f"gene@{geneset_name}",
            entropy_weight=entropy_weight,
            con_entropy_weight=con_entropy_weight,
            seed=seed
        )
        self._set_seed()
        self.geneset_name = geneset_name
        # 设置训练参数
        self.pretrain_epochs = 100
        self.train_epochs = 200
        self.pretrain_lr = 1e-4
        self.train_lr = 1e-5
        
        # 编码器
        self.fc1 = nn.Linear(input_dim, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000, affine=False)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(num_features=1000, affine=False)
        self.fc3 = nn.Linear(1000, 4000)
        self.bn3 = nn.BatchNorm1d(num_features=4000, affine=False)
        self.fc4_mu = nn.Linear(4000, latent_dim)
        self.bn4_mu = nn.BatchNorm1d(num_features=latent_dim)
        self.fc4_logvar = nn.Linear(4000, latent_dim)
        self.bn4_logvar = nn.BatchNorm1d(num_features=latent_dim)

        # 解码器
        self.fc5 = nn.Linear(latent_dim, 4000)
        self.bn5 = nn.BatchNorm1d(num_features=4000, affine=False)
        self.fc6 = nn.Linear(4000, 1000)
        self.bn6 = nn.BatchNorm1d(num_features=1000, affine=False)
        self.fc7 = nn.Linear(1000, 1000)
        self.bn7 = nn.BatchNorm1d(num_features=1000, affine=False)
        self.fc8 = nn.Linear(1000, input_dim)
        self.bn8 = nn.BatchNorm1d(num_features=input_dim)
        
        self.pred = nn.Linear(latent_dim, num_clusters)
    
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
    
    def get_data_loaders(self, batch_size: int = 128, 
                        data_dir: str = 'Datasets',
                        n_input: int = 2000):
        """
        获取基因数据加载器
        Args:
            batch_size: 批次大小
            data_dir: 数据目录
            n_input: 高变基因数量
        """
        # 创建数据集
        dataset = GenomeDataset(data_dir, self.geneset_name, n_input)
        
        # 配置数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
        )
        
        return train_loader, test_loader 
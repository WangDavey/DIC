import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from base_dvc import BaseDVC
import os

class STL10Dataset(Dataset):
    def __init__(self, data_dir='Datasets'):
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(root_dir, data_dir)
        self.data = loadmat(os.path.join(self.data_dir, 'stl10_features_allflip_addtestp_BICUBIC.mat'))
    
    def __getitem__(self, index):
        img = torch.tensor(self.data['fea'][index]).to(torch.float32)
        lab = self.data['true'].reshape(-1)[index]
        return img, lab
    
    def __len__(self):
        return len(self.data['fea'])

class STL10DVC(BaseDVC):
    #def __init__(self, entropy_weight=128, con_entropy_weight=27,seed = 127):
    def __init__(self, entropy_weight, con_entropy_weight,seed):
        super(STL10DVC, self).__init__(
            input_dim=2048,  # STL10输入维度
            latent_dim=10,   # 潜在空间维度
            num_clusters=10,  # STL10类别数
            dataset_name="stl10",  # 数据集名称
            entropy_weight=entropy_weight,  # 熵损失权重
            con_entropy_weight=con_entropy_weight,  # 条件熵损失权重
            seed = seed
        )
        self._set_seed()
        # 设置训练参数
        self.pretrain_epochs = 100  # 预训练轮数
        self.train_epochs = 100     # 训练轮数
        self.pretrain_lr = 1e-4     # 预训练学习率
        self.train_lr = 1e-5        # 训练学习率
        
        
        # 编码器
        self.fc1 = nn.Linear(2048, 500)
        self.bn1 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc3 = nn.Linear(500, 2000)
        self.bn3 = nn.BatchNorm1d(num_features=2000, affine=False)
        self.fc4_mu = nn.Linear(2000, 10)
        self.bn4_mu = nn.BatchNorm1d(num_features=10)
        self.fc4_logvar = nn.Linear(2000, 10)
        self.bn4_logvar = nn.BatchNorm1d(num_features=10)

        # 解码器
        self.fc5 = nn.Linear(10, 2000)
        self.bn5 = nn.BatchNorm1d(num_features=2000, affine=False)
        self.fc6 = nn.Linear(2000, 500)
        self.bn6 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc7 = nn.Linear(500, 500)
        self.bn7 = nn.BatchNorm1d(num_features=500, affine=False)
        self.fc8 = nn.Linear(500, 2048)
        self.bn8 = nn.BatchNorm1d(num_features=2048)
        
        self.pred = nn.Linear(10, 10)
    
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
    
    # def loss_function(self, mu, logvar, pred,recon_x,x):
    #     """
    #     recon_x: generating images
    #     x: origin images
    #     mu: latent mean
    #     logvar: latent log variance
    #     """
    #     import math
    #     pred_weight = self.state_dict()['pred.weight']
    #     #print(pred_weight.shape)
    #     element_1 = mu.unsqueeze(1)
    #     element_2 = pred_weight.unsqueeze(0)
    #     mu_v = element_1 - element_2
    #     logvar_v = logvar.unsqueeze(1)
    #     pred_v = pred.unsqueeze(2)
    #     KLD_element = torch.sum(mu_v.pow(2).mul(pred_v) + logvar_v.exp(), dim = 1)
    #     KLD_element = KLD_element + logvar.exp() + math.log(math.pi)
    #     KLD = torch.sum(KLD_element).mul_(0.5)
    #     f_j = torch.sum(pred, axis=0)
    #     entropy = torch.sum(torch.mul(f_j, torch.log(f_j)))
    #     con_entropy = -1 * torch.sum(torch.mul(pred, torch.log(pred + 1.0e-8)))
    #     return KLD + self.con_entropy_weight * con_entropy + self.entropy_weight * entropy
    
    def _build_encoder(self):
        """这个方法不再使用，但需要实现以满足父类要求"""
        return None, None, None
    
    def _build_decoder(self):
        """这个方法不再使用，但需要实现以满足父类要求"""
        return None
    
    @staticmethod
    def get_data_loaders(batch_size=128, data_dir='Datasets'):
        """获取STL10数据加载器"""
        # 创建数据集
        dataset = STL10Dataset(data_dir)
        
        # 创建数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,  # 使用多进程加载
            pin_memory=True,  # 使用固定内存
            persistent_workers=True  # 保持工作进程存活
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=13000,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        
        return train_loader, test_loader 
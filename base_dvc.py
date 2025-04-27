import os
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import random

class BaseDVC(nn.Module):
    def __init__(self, input_dim, latent_dim, num_clusters, dataset_name, entropy_weight=0, con_entropy_weight=0, seed = 0):
        super(BaseDVC, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.dataset_name = dataset_name
        
        # 损失函数权重
        self.entropy_weight = entropy_weight
        self.con_entropy_weight = con_entropy_weight
        
        # 训练参数
        self.pretrain_epochs = 0  # 预训练轮数
        self.train_epochs = 0     # 训练轮数
        self.pretrain_lr = 0.0    # 预训练学习率
        self.train_lr = 0.0       # 训练学习率
        self.seed = seed             # 随机种子
        
        # 创建保存目录
        self.pretrain_dir = f"./pretrain_models/{dataset_name.lower()}"
        self.results_dir = f"./results/{dataset_name.lower()}"
        os.makedirs(self.pretrain_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def get_params(self):
        """获取模型参数"""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'num_clusters': self.num_clusters,
            'dataset_name': self.dataset_name,
            'entropy_weight': self.entropy_weight,
            'con_entropy_weight': self.con_entropy_weight,
            'pretrain_epochs': self.pretrain_epochs,
            'train_epochs': self.train_epochs,
            'pretrain_lr': self.pretrain_lr,
            'train_lr': self.train_lr,
            'seed': self.seed
        }
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_()
        device = mu.device
        eps = eps.to(device)
        return eps.mul(std).add_(mu)
    
    def loss_function(self, mu, logvar, pred,recon_x,x):
        """计算损失函数"""
        pred_weight = self.state_dict()['pred.weight']
        element_1 = mu.unsqueeze(1)
        element_2 = pred_weight.unsqueeze(0)
        mu_v = element_1 - element_2
        logvar_v = logvar.unsqueeze(1)
        pred_v = pred.unsqueeze(2)
        
        KLD_element = torch.sum(mu_v.pow(2).mul(pred_v) + logvar_v.exp(), dim=1)
        KLD_2 = torch.sum(element_2.pow(2).mul(pred_v), dim=1) * (-2)
        KLD_3 = torch.sum(element_2.mul(pred_v), dim=1).pow(2) * 2
        
        KLD = torch.sum(KLD_element + KLD_2 + KLD_3).mul_(0.5)
        
        f_j = torch.sum(pred, axis=0)
        entropy = torch.sum(torch.mul(f_j, torch.log(f_j)))
        con_entropy = -1 * torch.sum(torch.mul(pred, torch.log(pred + 1.0e-8)))
        
        # reconstruction_function = nn.MSELoss(reduction='sum')
        # MSE = reconstruction_function(recon_x, x)

        return KLD,con_entropy,entropy, KLD + self.con_entropy_weight * con_entropy + self.entropy_weight * entropy
    
    def pretrain_loss_function(self, recon_x, x, mu, logvar):
        """预训练时的损失函数"""
        reconstruction_function = nn.MSELoss(reduction='sum')
        MSE = reconstruction_function(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE + KLD
    
    def train_step(self, dataloader, optimizer):
        """单步训练"""
        self.train()
        train_loss = 0.0
        for x, _ in dataloader:
            x = x.cuda()
            #x = x.to(device)
            mu, logvar, pred, z = self(x)
            # recon_x = self.decoder(z)
            _,_,_,loss = self.loss_function(mu, logvar, pred,None,x)
            loss = loss / x.shape[0]
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return train_loss / len(dataloader.dataset)
    
    def pretrain_step(self, x, optimizer):
        """预训练单步"""
        self.train()
        x = x.cuda()
        #x = x.to(device)
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decoder(z)
        loss = self.pretrain_loss_function(recon_x, x, mu, logvar) / x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def initialize_pred_with_kmeans(self, dataloader):
        """使用KMeans中心点初始化预测层"""
        self.eval()
        all_features = []
        all_y = []
        # 收集所有特征
        with torch.no_grad():
            for x, y in dataloader:
                x = x.cuda()
                mu, _, _, _ = self(x)
                all_features.append(mu.cpu().numpy())
                all_y.append(y)
        
        all_features = np.vstack(all_features)
        all_y = np.hstack(all_y)
        # 使用KMeans获取中心点
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed, n_init=10)
        label_cluster = kmeans.fit(all_features).labels_
        acc = 1 - self.err_rate(y.numpy(), label_cluster)
        print(f"acc: {acc}")
        # 使用KMeans中心点初始化预测层权重
        with torch.no_grad():
            self.pred.weight.data = torch.from_numpy(kmeans.cluster_centers_).float().cuda()
            self.pred.bias.data.zero_()
        
        return kmeans.cluster_centers_
        
    def evaluate_pretrain(self, dataloader):
        self.eval()
        val_loss = 0.0
        total_acc = 0.0
        total_time = 0.0
        fla = 1
        for x, y in dataloader:
            time_start = time.time()
            with torch.no_grad():
                im = x.cuda()
                mu, logvar= self.encoder(im)
                recon_x = self.decoder(mu)
                end_time = time.time()
                total_time += (end_time - time_start)
                features = KMeans(n_clusters=self.num_clusters, random_state=0,n_init=10).fit(mu.cpu())
                label_cluster = features.labels_
                if fla == 1:
                    y_pred = label_cluster
                    #y_pred = label_cluster
                    y_true = y.numpy()
                    y_mu = mu.cpu().numpy()
                    fla = 2
                else:
                    y_pred = np.hstack((y_pred, label_cluster))
                    #y_pred = np.hstack((y_pred, label_cluster))
                    y_true = np.hstack((y_true, y.numpy()))
                    y_mu = np.vstack((y_mu, mu.cpu().numpy()))
        missrate_x = self.err_rate(y_true, y_pred)
        acc = 1 - missrate_x
        #nmi = NMI(y.numpy(), label_cluster.cpu().numpy())
        nmi = NMI(y_true, y_pred)
        print(f"total_time: {total_time}")
        return acc, nmi, mu.cpu(), label_cluster, y.numpy()
        
    def evaluate(self, dataloader):
        """评估模型性能"""
        self.eval()
        val_loss = 0.0
        total_acc = 0.0
        fla = 1
        for x, y in dataloader:
            time_start = time.time()
            with torch.no_grad():
                im = x.cuda()
                mu, _ = self.encoder(im)
                pred = F.softmax(self.pred(mu), dim=1)
                mu, logvar, pred, sampling = self(im)
                recon_x = self.decoder(sampling)
                KLD_loss,con_entropy_loss,entropy_loss,loss = self.loss_function(mu, logvar, pred,recon_x,im)
                KLD_loss = KLD_loss / im.shape[0]
                con_entropy_loss = con_entropy_loss / im.shape[0]
                entropy_loss = entropy_loss / im.shape[0]
                loss = loss / im.shape[0]
                # features = KMeans(n_clusters=self.num_clusters, random_state=0).fit(sampling.cpu())
                # label_cluster_kmeans = features.labels_
                
                label_cluster = pred.argmax(1)
                if fla == 1:
                    y_pred = label_cluster.cpu().numpy()
                    #y_pred = label_cluster
                    y_true = y.numpy()
                    y_mu = mu.cpu().numpy()
                    pred = pred.cpu().numpy()
                    fla = 2
                else:
                    y_pred = np.hstack((y_pred, label_cluster.cpu().numpy()))
                    #y_pred = np.hstack((y_pred, label_cluster))
                    y_true = np.hstack((y_true, y.numpy()))
                    y_mu = np.vstack((y_mu, mu.cpu().numpy()))
                    pred = np.vstack((pred, pred.cpu().numpy()))
                # missrate_x = self.err_rate(y.numpy(), label_cluster.cpu().numpy())
                # acc = 1 - missrate_x
                # nmi = NMI(y.numpy(), label_cluster.cpu().numpy())
                # missrate_x_kmeans = self.err_rate(y.numpy(), label_cluster_kmeans)
                # acc_kmeans = 1 - missrate_x_kmeans
                # nmi_kmeans = NMI(y.numpy(), label_cluster_kmeans)
                # print(f"acc_kmeans:{acc_kmeans}, nmi_kmeans:{nmi_kmeans}")
                
                #print(mu.cpu())
                # total_acc +=acc
                # print(len(Img.dataset))
                #print("epoch: %d" % iter_ft, "cost: %.8f" % Loss, "acc: %.4f" % acc)
                # print(label_cluster.cpu().shape)
                # print(label_cluster.cpu())

        loss_dict = {
            'KLD_loss': KLD_loss,
            'con_entropy_loss': con_entropy_loss,
            'entropy_loss': entropy_loss,
            'loss': loss
        }
        missrate_x = self.err_rate(y_true, y_pred)
        acc = 1 - missrate_x
        #nmi = NMI(y.numpy(), label_cluster.cpu().numpy())
        nmi = NMI(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        # 计算purity
        purity = 0.0
        for i in range(self.num_clusters):
            cluster_indices = np.where(y_pred == i)[0]
            if len(cluster_indices) > 0:
                cluster_labels = y_true[cluster_indices]
                majority = np.bincount(cluster_labels).max()
                purity += majority
        purity = purity / len(y_true)
        
        return acc, nmi, ari, purity, mu.cpu(), pred, y.numpy(), loss_dict, y_pred, y_true, y_mu

    @staticmethod
    def best_map(L1, L2):
        """计算最佳映射"""
        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        
        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        newL2 = np.zeros(L2.shape)
        for i in range(nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]
        return newL2
    
    @staticmethod
    def err_rate(gt_s, s):
        """计算错误率"""
        c_x = BaseDVC.best_map(gt_s, s)
        err_x = np.sum(gt_s[:] != c_x[:])
        missrate = err_x.astype(float) / (gt_s.shape[0])
        return missrate 
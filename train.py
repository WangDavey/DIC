import os
import torch
import torch.optim as optim
import argparse
from model_factory import ModelFactory
import random
import numpy as np
from scipy.io import savemat
from utils.lr_manager import CyclicalIntervalLRManager

def pretrain_model(model, train_loader, test_loader, num_epochs=None, acc_threshold=0.5, patience=10):
    """预训练模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=model.pretrain_lr)
    
    print(f"开始对数据集{model.dataset_name}预训练...")
    best_acc = 0
    best_nmi = 0
    best_model_state = None
    patience_counter = 0
    
    # 使用模型中的预训练轮数，如果没有指定
    if num_epochs is None:
        num_epochs = model.pretrain_epochs
    
    for epoch in range(num_epochs):
        # 使用模型中的pretrain_step方法
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            loss = model.pretrain_step(data, optimizer)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估模型
        acc, nmi, _, _, _ = model.evaluate_pretrain(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"预训练 Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, ACC: {acc:.4f}, NMI: {nmi:.4f}")
        
        # 早停检查
        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"当前最佳ACC: {acc:.4f}, NMI: {nmi:.4f}")
        else:
            patience_counter += 1
        
        # # 如果达到阈值或超过耐心值，停止训练
        # if patience_counter >= patience:
        #     print(f"预训练提前停止！最终ACC: {acc:.4f}, NMI: {nmi:.4f}")
        #     break
    
    # 训练结束后，保存最佳模型
    if best_model_state is not None:
        pretrain_path = os.path.join(model.pretrain_dir, f'pretrained_model.pth')
        torch.save(best_model_state, pretrain_path)
        print(f"保存最佳预训练模型，ACC: {best_acc:.4f}, NMI: {best_nmi:.4f}")
    
    print(f"预训练完成！最佳ACC: {best_acc:.4f}, NMI: {best_nmi:.4f}")
    return best_acc, best_nmi

def train_model(model, train_loader, test_loader, batch_size=128):
    """训练模型的主函数"""
    # 创建模型
    #model = ModelFactory.create_model(model_name)
    model_name = model.dataset_name
    params = model.get_params()
    print(f"模型参数: {params}")
      
    # 设置设备
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    
    # 训练阶段
    print(f"\n开始训练 {model_name} 模型...")
    optimizer = optim.Adam(model.parameters(), lr=model.train_lr)
   
    for epoch in range(model.train_epochs):
        # 使用模型中的train_step方法
        train_loss = model.train_step(train_loader, optimizer)
        
        # 评估模型
        acc, nmi,ari,purity, all_features, all_preds, all_labels,loss,_,_,_ = model.evaluate(test_loader)
        print("epoch: %d" % epoch, "acc: %.4f" % acc, "nmi: %.4f" % nmi, "ari: %.4f" % ari, "purity: %.4f" % purity)
        print("loss:",loss)
    print(f"\n{model_name} 模型评估结果:")
    print(f"准确率: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")
    

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DVC模型训练脚本')
    # parser.add_argument('--dataset', type=str, default='gene@Quake_10x_Limb_Muscle',
    #                   help='数据集名称 (默认: cifar10)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='数据集名称 (默认: cifar10)')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU设备编号 (默认: 0)')
    parser.add_argument('--pretrain', default= False,type = bool,
                      help='是否进行预训练')
    parser.add_argument('--acc_threshold', type=float, default=0.5,
                      help='预训练ACC阈值 (默认: 0.5)')
    parser.add_argument('--patience', type=int, default=10,
                      help='预训练早停耐心值 (默认: 10)')
    parser.add_argument('--use_kmeans', action='store_true',
                      help='是否使用KMeans初始化预测 (默认: False)')
    
    args = parser.parse_args()
    
    # 创建模型以获取随机种子
    model = ModelFactory.create_model(args.dataset)
    
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.cuda.set_device(device)
    # 将模型移到设备上
    model = model.to(device)
    
    # 获取数据加载器
    train_loader, test_loader = model.get_data_loaders()
    
    pretrain_dir = model.pretrain_dir
    pretrain_path = os.path.join(pretrain_dir, "pretrained_model.pth")
    # 预训练模型
    if args.pretrain or not os.path.exists(pretrain_path):
        pretrain_model(model, train_loader, test_loader, 
                      acc_threshold=args.acc_threshold,
                      patience=args.patience)
    else:
    # 加载最佳预训练模型
        if os.path.exists(pretrain_dir):
            model.load_state_dict(torch.load(pretrain_path), strict=False)
            if args.use_kmeans:
                model.initialize_pred_with_kmeans(test_loader)

    
    # 设置保存目录
    save_dir = os.path.join("results", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练模型
    train_model(model,train_loader,test_loader)
    
    # 保存最终结果
    final_results_path = os.path.join(save_dir, "final_results.mat")
    acc, nmi, ari, purity, features, pred, labels,loss,y_pred,y_true,_ = model.evaluate(test_loader)


if __name__ == "__main__":
    main() 
# DIC (Deep Image Clustering)

DIC 是一个基于深度学习的图像聚类框架，支持多种数据集和模型架构。

## 特性

- 支持多种数据集：MNIST、FashionMNIST、CIFAR-10、STL-10、Mini-ImageNet 等
- 灵活的模型架构配置
- 预训练模型支持
- 可扩展的代码结构

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/WangDavey/DIC.git
cd DIC
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集：
   - 将数据集放在 `Datasets` 目录下
   - 支持的数据集格式包括：.mat、.npy、.pth 等

2. 训练模型：
```bash
python train.py --dataset mnist --model_type dvc
```

3. 使用预训练模型：
   - 预训练模型存储在 `pretrain_models` 目录下
   - 可以通过 `--pretrained` 参数指定预训练模型路径

## 项目结构

```
DIC/
├── Datasets/           # 数据集目录
├── pretrain_models/    # 预训练模型目录
├── utils/             # 工具函数
├── base_dvc.py        # 基础模型类
├── train.py           # 训练脚本
├── requirements.txt   # 依赖列表
└── README.md          # 项目说明
```

## 注意事项

- 数据集和预训练模型文件较大，已通过 .gitignore 排除
- 首次运行需要下载相应的数据集
- 建议使用 GPU 进行训练

## 许可证

MIT License

## 联系方式

- 邮箱：hyper88@163.com 
from cifar10_dvc import Cifar10DVC
from fmnist_dvc import FMNISTDVC
from hhar_dvc import HHARDVC
from mnist_dvc import MNISTDVC
from reuters_dvc import ReutersDVC
from stl10_dvc import STL10DVC
from gene_dvc import GenomeDVC, GENESET_CONFIGS
from mini_ImageNet_dvc import MiniImageNetDVC
# 后续可以导入其他数据集的模型

class ModelFactory:
    @staticmethod
    def create_model(dataset_name, entropy_weight=None, con_entropy_weight=None, seed = None):
        """
        根据数据集名称创建对应的模型实例
        
        Args:
            dataset_name (str): 数据集名称
            entropy_weight (float, optional): 熵损失权重
            con_entropy_weight (float, optional): 条件熵损失权重
            seed (int, optional): 随机种子
            
        Returns:
            BaseDVC: 创建的模型实例
        """
        # 检查是否是基因数据集
        if dataset_name.startswith('gene@'):
            geneset_name = dataset_name.split('@')[1]
            if geneset_name not in GENESET_CONFIGS:
                raise ValueError(f"不支持的基因数据集: {geneset_name}")
            model_class = GenomeDVC
            kwargs = {'geneset_name': geneset_name}
        else:
            model_map = {
                'cifar10': Cifar10DVC,
                'fmnist': FMNISTDVC,
                'hhar': HHARDVC,
                'mnist': MNISTDVC,
                'reuters': ReutersDVC,
                'stl10': STL10DVC,
                'mini_imagenet': MiniImageNetDVC
            }
            
            if dataset_name.lower() not in model_map:
                raise ValueError(f"不支持的数据集: {dataset_name}")
                
            model_class = model_map[dataset_name.lower()]
            kwargs = {}
        
        # 如果提供了权重参数，则使用提供的值
        if entropy_weight is not None and con_entropy_weight is not None:
            kwargs.update({
                'entropy_weight': entropy_weight,
                'con_entropy_weight': con_entropy_weight
            })
        if seed is not None:
            kwargs['seed'] = seed
            
        return model_class(**kwargs) 
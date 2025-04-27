import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

class IntervalLRManager:
    """
    间歇性评估的学习率管理器
    
    特点：
    1. 完全独立于PyTorch的调度器接口
    2. 专为间歇性评估设计
    3. 支持可视化和日志记录
    """
    def __init__(self, 
                 optimizer, 
                 config=None,
                 initial_lr=None,
                 min_lr=None,
                 max_lr=None,
                 warmup_steps=None,
                 factor=0.5,
                 patience=5,
                 threshold=1e-4,
                 significant_threshold=0.01,
                 increase_factor=1.05,
                 early_stop_min_progress=0.2,
                 verbose=True):
        """
        初始化学习率管理器
        
        参数:
        - optimizer: PyTorch优化器
        - config: 配置对象（可选），包含学习率相关配置
        - initial_lr: 初始学习率，如未指定则使用optimizer当前值
        - min_lr: 最小学习率
        - max_lr: 最大学习率
        - warmup_steps: 预热步数
        - factor: 学习率衰减因子
        - patience: 容忍评估次数
        - threshold: 普通改进阈值
        - significant_threshold: 显著改进阈值
        - increase_factor: 学习率增加因子
        - early_stop_min_progress: 早停所需的最小进度比例
        - verbose: 是否输出详细信息
        """
        self.optimizer = optimizer
        self.config = config
        self.verbose = verbose
        
        # 从config或参数中获取配置
        self.base_lr = initial_lr if initial_lr is not None else optimizer.param_groups[0]['lr']
        
        # 如果config中有相关配置，优先使用config
        if config:
            self.min_lr = getattr(config, 'min_lr', self.base_lr * 0.0001) if min_lr is None else min_lr
            self.max_lr = getattr(config, 'max_lr', self.base_lr * 2.0) if max_lr is None else max_lr
            self.warmup_steps = getattr(config, 'warmup_steps', 100) if warmup_steps is None else warmup_steps
            self.total_steps = getattr(config, 'train_epochs', 1000) * getattr(config, 'steps_per_epoch', 1)
        else:
            self.min_lr = self.base_lr * 0.0001 if min_lr is None else min_lr
            self.max_lr = self.base_lr * 2.0 if max_lr is None else max_lr
            self.warmup_steps = 100 if warmup_steps is None else warmup_steps
            self.total_steps = 1000  # 默认总步数
        
        # 学习率调整参数
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.significant_threshold = significant_threshold
        self.increase_factor = increase_factor
        self.early_stop_min_progress = early_stop_min_progress
        
        # 状态追踪
        self.current_lr = self.base_lr * 0.1  # 从低学习率开始预热
        self.best_score = None
        self.best_epoch = None
        self.current_step = 0
        self.bad_counter = 0
        self.last_eval_step = 0
        
        # 历史记录
        self.lr_history = []
        self.step_history = []
        self.score_history = []
        self.eval_steps = []
        
        # 设置初始学习率
        self._set_lr(self.current_lr)
        
        if self.verbose:
            print(f"初始化学习率管理器: 初始lr={self.current_lr:.6f}, "
                  f"最小lr={self.min_lr:.6f}, 最大lr={self.max_lr:.6f}, "
                  f"预热步数={self.warmup_steps}")
    
    def step(self, metrics=None):
        """
        执行一步学习率调整
        
        参数:
        - metrics: 如果提供，表示这是一个评估步；否则是普通训练步
        
        返回:
        - 当前学习率
        """
        self.current_step += 1
        self.step_history.append(self.current_step)
        
        # 预热阶段
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            self.current_lr = self.base_lr * progress
            self._set_lr(self.current_lr)
            self.lr_history.append(self.current_lr)
            return self.current_lr
        
        # 记录当前学习率
        self.lr_history.append(self.current_lr)
        
        # 如果没有提供评估指标，仅记录学习率
        if metrics is None:
            return self.current_lr
        
        # 以下是评估步逻辑
        self.eval_steps.append(self.current_step)
        self.score_history.append(metrics)
        self.last_eval_step = self.current_step
        
        is_best = False
        
        # 首次评估或有改进
        if self.best_score is None or metrics > self.best_score + self.threshold:
            improvement = 0 if self.best_score is None else metrics - self.best_score
            self.best_score = metrics
            self.best_epoch = self.current_step
            is_best = True
            self.bad_counter = 0
            
            # 如果是显著改进且在训练前期，可适度增加学习率
            if (improvement > self.significant_threshold and 
                self.current_step < self.total_steps * 0.5):
                old_lr = self.current_lr
                self.current_lr = min(self.current_lr * self.increase_factor, self.max_lr)
                self._set_lr(self.current_lr)
                
                if self.verbose and old_lr != self.current_lr:
                    print(f"显著改进 (+{improvement:.4f})，"
                          f"增加学习率: {old_lr:.6f} -> {self.current_lr:.6f}")
        else:
            # 无改进
            self.bad_counter += 1
            
            # 如果连续多次无改进，降低学习率
            if self.bad_counter >= self.patience:
                # 计算训练进度
                progress_ratio = self.current_step / self.total_steps
                
                # 根据训练进度调整衰减强度
                adjusted_factor = self.factor
                if progress_ratio > 0.7:  # 训练后期
                    adjusted_factor = self.factor * 0.8  # 更激进的衰减
                
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * adjusted_factor, self.min_lr)
                self._set_lr(self.current_lr)
                self.bad_counter = 0  # 重置计数器
                
                if self.verbose:
                    print(f"连续 {self.patience} 次评估无改进，"
                          f"降低学习率: {old_lr:.6f} -> {self.current_lr:.6f}")
        
        return self.current_lr
    
    def should_stop_early(self):
        """判断是否应该早停"""
        if self.best_epoch is None:
            return False
            
        # 如果学习率已接近最小值且长时间无改进
        if self.current_lr <= self.min_lr * 1.1:
            steps_since_best = self.current_step - self.best_epoch
            min_progress = self.total_steps * self.early_stop_min_progress
            
            if steps_since_best > min_progress:
                if self.verbose:
                    print(f"触发早停: 学习率={self.current_lr:.6f} 接近最小值，"
                          f"且 {steps_since_best} 步无改进")
                return True
                
        return False
    
    def _set_lr(self, lr):
        """设置优化器的学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def add_lr_noise(self, epoch, noise_range=0.05, noise_prob=0.1):
        """
        以一定概率对学习率添加随机扰动
        
        参数:
        - epoch: 当前训练轮次
        - noise_range: 噪声范围，相对于当前学习率的比例
        - noise_prob: 添加噪声的概率
        """
        # 每隔一定间隔考虑添加噪声
        if epoch % 20 == 0:  # 每20个epoch考虑一次
            # 以noise_prob的概率添加噪声
            if random.random() < noise_prob:
                # 生成范围为[-noise_range, +noise_range]的随机扰动
                noise = (1.0 + (random.random() * 2 - 1) * noise_range)
                old_lr = self.current_lr
                # 应用扰动，但确保在允许范围内
                self.current_lr = min(max(self.current_lr * noise, self.min_lr), self.max_lr)
                self._set_lr(self.current_lr)
                
                if self.verbose:
                    print(f"在epoch {epoch}添加学习率扰动: {old_lr:.6f} -> {self.current_lr:.6f}")
                
                return True
        return False
    
    def plot(self, save_dir=None):
        """绘制学习率和性能曲线"""
        if not save_dir:
            save_dir = './results'
            
        import os
        os.makedirs(save_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 绘制学习率历史
        plt.figure(figsize=(12, 6))
        plt.plot(self.step_history, self.lr_history, 'b-', linewidth=1, alpha=0.5)
        
        # 如果有评估点，标记它们
        if self.eval_steps:
            eval_lrs = [self.lr_history[self.step_history.index(step)] for step in self.eval_steps]
            plt.plot(self.eval_steps, eval_lrs, 'ro', markersize=4)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        lr_path = f"{save_dir}/lr_history_{timestamp}.png"
        plt.savefig(lr_path)
        plt.close()
        
        # 2. 绘制评估指标历史
        if self.score_history:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Learning Rate', color=color)
            ax1.plot(self.step_history, self.lr_history, color=color, alpha=0.5)
            ax1.scatter(self.eval_steps, 
                      [self.lr_history[self.step_history.index(step)] for step in self.eval_steps], 
                      color=color, s=20)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_yscale('log')
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Evaluation Metric', color=color)
            ax2.plot(self.eval_steps, self.score_history, 'o-', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 标记最佳点
            if self.best_score is not None:
                best_idx = self.score_history.index(self.best_score)
                best_step = self.eval_steps[best_idx]
                ax2.scatter([best_step], [self.best_score], color='green', s=100, marker='*')
            
            fig.tight_layout()
            plt.title('Learning Rate vs Evaluation Metric')
            plt.grid(True, alpha=0.3)
            
            metric_path = f"{save_dir}/metrics_{timestamp}.png"
            plt.savefig(metric_path)
            plt.close()
            
        return [lr_path, metric_path if self.score_history else None]
    

class CyclicalIntervalLRManager(IntervalLRManager):
    """
    周期性学习率管理器 - 扩展基本的间歇性学习率管理器，添加周期性策略
    """
    def __init__(self, 
                 optimizer, 
                 config=None,
                 cycle_length=500,      # 一个完整周期的步数
                 cycle_mult=1.0,        # 每个周期后的长度倍增因子
                 initial_lr=None,      # 统一使用父类参数名
                 max_lr=None,
                 **kwargs):
        super().__init__(optimizer, config, initial_lr=initial_lr, **kwargs)
        
        # 周期性学习率参数
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.base_lr = self.current_lr  # 直接使用父类初始化后的current_lr
        self.max_lr = self.max_lr if max_lr is None else max_lr
        
        # 周期状态追踪
        self.cycle_count = 0
        self.cycle_step = 0
        self.use_cyclical = True  # 是否启用周期性调整
        
    def step(self, metrics=None):
        """重写step方法，添加周期性学习率调整"""
        self.current_step += 1
        self.cycle_step += 1
        self.step_history.append(self.current_step)
        
        # 检查是否完成一个周期
        current_cycle_length = self.cycle_length * (self.cycle_mult ** self.cycle_count)
        if self.cycle_step >= current_cycle_length and self.use_cyclical:
            self.cycle_step = 0
            self.cycle_count += 1
            
            # 周期性地提高学习率，帮助逃离局部最优
            old_lr = self.current_lr
            self.current_lr = self.max_lr / (2 ** (self.cycle_count // 3))  # 逐渐降低峰值
            self._set_lr(self.current_lr)
            
            if self.verbose:
                print(f"完成周期 {self.cycle_count}，重置学习率: {old_lr:.6f} -> {self.current_lr:.6f}")
        
        # 如果在预热阶段
        elif self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            self.current_lr = self.base_lr * progress
            self._set_lr(self.current_lr)
        
        # 记录当前学习率
        self.lr_history.append(self.current_lr)
        
        # 处理评估步骤
        if metrics is not None:
            # 评估时，禁用周期性调整，转为自适应调整
            self.use_cyclical = False
            
            # 调用父类的评估逻辑
            return super().step(metrics)
            
        return self.current_lr

# Organo

Organo 是一个基于 PyTorch Lightning 构建的模块化深度学习训练框架，提供灵活的配置管理和组件注册系统。

## ✨ 特性

- 🔌 **模块化注册系统**：支持模型、任务、损失函数、优化器和日志记录器的统一注册和管理
- ⚙️ **灵活的配置管理**：基于 OmegaConf 和 Hydra 的配置系统，支持配置合并和覆盖
- 📊 **多种日志记录器**：内置支持 WandB、TensorBoard 和 CSV 日志记录
- 🚀 **Lightning 集成**：基于 PyTorch Lightning，轻松实现分布式训练和混合精度训练
- 📦 **数据模块抽象**：简化数据加载和预处理流程

## 📋 环境要求

- Python >= 3.10, < 3.13
- PyTorch
- PyTorch Lightning

## 🔧 安装

### 使用 Poetry 安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/MigoXV/organo.git
cd organo

# 使用 Poetry 安装依赖
poetry install
```

### 使用 pip 安装

```bash
pip install organo
```

## 🚀 快速开始

### 注册自定义模型

```python
import torch
from dataclasses import dataclass
from organo.registers import model_registry

@dataclass
class MyModelConfig:
    hidden_size: int = 256
    num_layers: int = 4

@model_registry.register("my_model", MyModelConfig)
class MyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # 模型初始化
        pass
```

### 注册自定义任务

```python
from organo.registers import task_registry
from organo.tasks.fsmn import FSMNVAD

@task_registry.register("my_task")
class MyTask(FSMNVAD):
    def training_step(self, batch, batch_idx):
        # 训练逻辑
        pass
    
    def validation_step(self, batch, batch_idx):
        # 验证逻辑
        pass
```

### 配置训练

```python
from omegaconf import OmegaConf
from organo.train import train

config = OmegaConf.create({
    "meta": {
        "model": "my_model",
        "task": "my_task",
        "criterion": "cross_entropy",
        "logger": "wandb"
    },
    "model": {
        "hidden_size": 512,
        "num_layers": 6
    },
    "logger": {
        "project": "my_project",
        "name": "experiment_1"
    }
})

train(config)
```

## 📁 项目结构

```
organo/
├── src/
│   └── organo/
│       ├── configs/         # 配置数据类定义
│       │   ├── checkpoint.py    # 检查点配置
│       │   ├── config.py        # 主配置
│       │   ├── dataloader.py    # 数据加载器配置
│       │   ├── logger.py        # 日志记录器配置
│       │   └── trainer.py       # 训练器配置
│       ├── data/            # 数据处理模块
│       │   ├── datamodule.py    # Lightning DataModule
│       │   └── utils/           # 数据工具函数
│       ├── loggers/         # 日志记录器注册
│       ├── registries/      # 注册系统
│       │   └── module_registry.py  # 模块注册器
│       ├── tasks/           # 任务定义
│       │   └── fsmn.py          # FSMN VAD 任务基类
│       ├── registers.py     # 全局注册器实例
│       └── train.py         # 训练入口
├── tests/               # 测试文件
├── pyproject.toml       # 项目配置
└── README.md
```

## 📖 配置说明

### 日志记录器配置

#### WandB

```yaml
logger:
  name: experiment_name
  project: project_name
  save_dir: ./logs
  offline: false
```

#### TensorBoard

```yaml
logger:
  save_dir: ./logs
  name: lightning_logs
  log_graph: false
```

#### CSV

```yaml
logger:
  save_dir: ./logs
  name: lightning_logs
  flush_logs_every_n_steps: 100
```

### 检查点配置

```yaml
checkpoint:
  dirpath: ./checkpoints
  filename: "{epoch}-{val_loss:.2f}"
  monitor: val_loss
  mode: min
  save_top_k: 3
  save_last: true
```

### 训练器配置

```yaml
trainer:
  accelerator: auto
  devices: auto
  max_epochs: 100
  log_every_n_steps: 10
  gradient_clip_val: 1.0
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## VLM2Vec-V2 训练流程分析

### 🔍 **核心训练组件**

#### 1. **主训练脚本** (`train.py`)
- 这是训练的入口点
- 负责初始化模型、数据集、训练器等核心组件
- 支持分布式训练和断点续训

#### 2. **模型架构** (`src/model/model.py`)
- `MMEBModel` 类：核心模型类，基于Qwen2-VL等backbone
- 支持LoRA微调
- 实现了对比学习的编码和相似度计算

#### 3. **数据处理流程**
- **数据集配置** (`experiments/public/train/train_alltasks.yaml`)：定义了多模态数据集的混合训练策略
- **数据加载器** (`src/data/loader/mixed_dataset.py`)：实现了多数据集的交错采样
- **数据收集器** (`src/data/collator/train_collator.py`)：处理多模态输入（文本+图像+视频）

#### 4. **训练器** (`src/trainer.py`)
- `GradCacheLateProcessTrainer`：支持梯度缓存的训练器
- 实现了对比学习的训练循环
- 支持分布式训练

#### 5. **损失函数** (`src/loss.py`)
- `SimpleContrastiveLoss`：基础对比损失
- `DistributedContrastiveLoss`：分布式对比损失

### 🔧 **关键配置参数**

#### 训练脚本参数 (`train_v2-qwen2vl-2B.sh`)：
```bash
--lora --lora_r 16                    # LoRA配置
--model_name Qwen/Qwen2-VL-2B-Instruct # 模型backbone
--pooling eos --normalize True         # 池化策略
--temperature 0.02                     # 对比学习温度
--grad_cache True                      # 梯度缓存
--per_device_train_batch_size 128     # 批次大小
--learning_rate 5e-5                  # 学习率
--max_steps 5000                      # 训练步数
```

### 📖 **数据集结构**

训练使用了三种主要数据类型：
1. **图像-文本数据** (MMEB-train)：包含20+个图像理解任务
2. **视频-文本数据** (LLaVA-Hound)：视频描述和问答
3. **视觉文档数据** (Vidore, VisRAG)：文档理解任务

### 🚀 **复现训练的关键步骤**

#### 1. **环境准备**
```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据
# 下载MMEB-V2数据集到指定路径
```

#### 2. **数据准备**
- 下载MMEB-V2数据集到 `vlm2vec_train/MMEB-train/`
- 准备视频数据到 `vlm2vec_train/train_video_and_instruction/`
- 配置数据集路径在YAML文件中

#### 3. **训练启动**
```bash
# 修改训练脚本中的路径
bash experiments/public/train/train_v2-qwen2vl-2B.sh
```

### 🎯 **需要重点关注的模块**

1. **`src/model/model.py`** - 模型架构和对比学习实现
2. **`src/data/dataset/mmeb_dataset.py`** - 多模态数据处理
3. **`src/trainer.py`** - 训练循环和梯度缓存
4. **`experiments/public/train/train_alltasks.yaml`** - 数据集配置
5. **`src/loss.py`** - 对比损失函数

### 💡 **训练技巧**

1. **梯度缓存**：使用 `--grad_cache True` 来处理大批次训练
2. **交错采样**：通过 `interleave_batch_size` 控制不同数据集的采样比例
3. **LoRA微调**：使用LoRA来高效微调大模型
4. **分布式训练**：支持多GPU训练

### ⚠️ **注意事项**

1. **数据路径**：确保所有数据集路径正确配置
2. **内存管理**：使用梯度缓存和适当的批次大小
3. **模型检查点**：支持断点续训，使用 `--resume_from auto`
4. **监控训练**：使用WandB进行训练监控

这个代码库实现了一个完整的多模态对比学习训练框架，支持图像、视频和文档的统一嵌入学习。复现时需要重点关注数据处理、模型架构和训练策略这三个核心部分。
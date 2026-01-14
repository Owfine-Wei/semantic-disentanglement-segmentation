# Semantic Disentanglement Segmentation (SDS)
## 零、运行环境与配置

**本项目基于 PyTorch 和 MMSegmentation 框架。**

### 1. 基础环境

`torch==1.10.1+cu113`

`torchvision==0.11.2+cu113`

### 2. MMSeg 环境配置（需确保 torch 版本正确）

（1）安装 mmengine 和 mmcv

`pip install -U openmim`  
`mim install mmengine`  
`mim install mmcv==2.0.0`

（2）安装其他依赖库

`pip install opencv-python pillow matplotlib seaborn tqdm pytorch-lightning 'mmdet>=3.1.0' -i https://pypi.tuna.tsinghua.edu.cn/simple`

（3）下载 MMSegmentation 源码

`git clone https://github.com/open-mmlab/mmsegmentation.git`

（4）安装 MMSegmentation

`pip install -v -e .`

## 一、训练之前

### 1. 数据集准备

请确保已经正确下载 **CityScapes** 数据集，并使用官方提供的  
`cityscapesscripts` 工具生成训练所需的标签文件：

- `gtFine_labelTrainIds.png`

该文件是后续语义分割训练与评估所使用的标准标注格式。

---

### 2. `config.py` 全局配置说明

在开始训练之前，需要首先配置 `helpers` 目录下的 `config.py` 文件。  
该文件用于定义**训练过程中的全局变量**，一旦确定后，**不建议在训练过程中频繁修改**。

`config.py` 主要包含以下几个部分：

#### （1）文件路径配置

用于指定以下路径信息：

- 原始 CityScapes 数据集路径  
- 前景数据集路径  
- 背景数据集路径  
- 可视化结果的保存路径  
- 模型权重（Checkpoint）的存储路径  

#### （2）数据处理参数

包括：

- 训练阶段使用的随机裁剪尺寸  
- 图像标准化所使用的 RGB 均值与方差  

#### （3）CityScapes 数据集定义

包含与语义分割类别相关的核心信息：

- 语义分割类别数  
- 各类别对应的 `TrainIds`  
- 前景类别的 `TrainIds`  
- 背景类别的 `TrainIds`  
- 语义类别与 `TrainIds` 之间的映射字典  
- 原始图像分辨率设置  

#### （4）类擦除样本数量

- 默认值为 **1**
- 可根据实验需求进行调整，用于控制类擦除样本的生成数量

#### （5）分割类别 RGB 颜色表

- 采用 **CityScapes 官方提供的颜色表**
- 主要用于可视化
- **不建议修改**

---

### 3. 前景 / 背景数据集生成

在确认 `config.py` 中的 `FOREBACK_DATA_DIR` 路径设置正确后，在当前文件夹下  
直接运行以下脚本：

```bash
PYTHONPATH=. python helpers/forebackground_data_generator.py
```
脚本将自动在指定目录下生成：

- 前景数据集（Foreground）

- 背景数据集（Background）

供后续训练与独立评测使用。

**注：在运行脚本时，环境变量PYTHONPATH需始终为当前文件夹**

## 二、helpers 目录介绍

`helpers` 目录主要用于存放 **训练、评估与可视化过程中使用的辅助模块**，  
涵盖指标计算、损失函数封装、训练工具以及结果可视化等功能。

---

### 1. 指标计算模块

用于在模型验证阶段计算多种语义分割评估指标，统一在 `val.py` 中调用。

- **`calculate_pa_miou.py`**  
  - 计算 Pixel Accuracy（PA）与 mean Intersection over Union（mIoU）  
  - 用于评估模型整体分割性能

- **`calculate_saiou.py`**  
  - 计算前景 IoU（FIoU）、背景 IoU（BIoU）以及独立 IoU（SAIoU）  
  - 用于分析模型在前景 / 背景语义解耦方面的表现

---

### 2. 训练辅助模块

该部分脚本主要用于 **辅助模型训练过程的稳定性与可控性**。

- **`integrated_loss.py`**  
  - 提供统一的损失函数计算接口  
  - 包含基础分割损失及一致性损失（Consistency Loss）

- **`Aux_loss.py`**  
  - 对辅助损失进行封装的 Wrapper  
  - 用于控制辅助损失是否生效及其权重

- **`Logger.py`**  
  - 提供训练日志记录功能  
  - 支持训练过程中的关键信息输出与保存

- **`Warmup_scheduler.py`**  
  - 实现学习率预热（Warmup）策略  
  - 用于提升训练初期的稳定性

- **`set_seed.py`**  
  - 用于统一设置随机种子  
  - 保证实验结果的可复现性

---

### 3. 可视化模块

用于直观展示模型在不同数据集及语义条件下的分割效果。

- **`visualize_foreback_val.py`**  
  - 可视化模型在前景验证集与背景验证集上的分割结果  
  - 用于分析模型对前景 / 背景语义的建模能力

- **`visualize_val.py`**  
  - 可视化模型在原始验证集上的分割结果  
  - 用于整体性能展示

- **`show_erf.py`**  
  - 可视化模型在指定语义类别下的有效感受野（Effective Receptive Field, ERF）  
  - 用于分析模型的空间感知特性

---

### 4. 类擦除样本生成器

- **`classes_erased_samples_generator.py`**  
  - 用于生成类擦除（Class Erasing）样本  
  - 返回以下内容：
    - 擦除后的原始图像  
    - 擦除后的标注图  
    - 类擦除掩码图  

该模块为语义解耦训练提供关键数据支持。

---

### 5. 其他辅助工具

- **`delete_non_files.py`**  
  - 用于清理数据集中训练过程中不会使用到的冗余文件  
  - 可按需运行，不影响模型训练流程

---

### 6. 模型封装

该部分用于存放具体的模型定义与封装代码。

- **`fcn_model.py`**  
- **`segformer_model.py`**

统一通过 `get_model()` 函数返回训练所需的模型实例，  
便于在不同实验配置下进行模型切换。

## 三、重要脚本介绍

本节对项目中承担核心功能的脚本进行说明，涵盖数据构建、模型定义、训练与评估等关键流程。

---

### 1. `data_sds_cityscapes.py`

该脚本用于构建 Semantic Disentanglement Segmentation（SDS）所需的多种数据集形式，
是整个训练与评估流程中的数据入口模块。

#### 提供的数据集类

- `Origin_CityScapes`   
  原始 CityScapes 数据集，仅包含原始图像及其对应的语义标注。

- `FOREBACK_CityScapes`  
  前景数据集与背景数据集，用于前景 / 背景的独立训练与评测。

- `CSG_CityScapes`  
  类擦除（Class-wise Semantic Erasing）数据集，用于语义解耦训练。

- `NDA_CityScapes`  
  朴素数据增强数据集（Naive Data Augmentation）。

- `SDS_CityScapes`  
  对上述数据集进行统一封装的综合数据集，  
  SDS 为 Semantic Disentanglement Segmentation 的缩写。

---

#### `load_data` 函数说明

`
load_data(mode, split, csg_mode)
`

- **mode**（数据加载模式）

  - origin  
    返回原始 CityScapes 数据集

  - foreground  
    返回前景数据集

  - background  
    返回背景数据集

  - csg_only  
    返回类擦除样本

  - csg+origin  
    返回类擦除样本及其对应的原始样本

  - nda  
    返回朴素数据增强数据集  
    （原始数据 + 前景数据 + 背景数据）

- **split**（数据用途）

  - train / val / test

- **csg_mode**（类擦除模式，仅在 csg_only 或 csg+origin 时生效）

  - foreground  
    仅随机擦除前景类

  - background  
    仅随机擦除背景类

  - both  
    前景类与背景类均可能被随机擦除

---

### 2. `models.py`

该脚本用于存放基于 MMSegmentation 框架的多种语义分割模型定义。

- 对不同模型结构进行统一封装
- 便于在不同实验设置下快速切换模型

通过 get_model 函数加载并返回训练所需的模型实例。

---

### 3. `val.py`

模型验证脚本，用于在验证集或测试集上评估模型性能。

主要计算的指标包括：

- **Pixel Accuracy**（PA）
- **mean Intersection over Union**（mIoU）
- **Foreground IoU**（FIoU）
- **Background IoU**（BIoU）
- **Stand-Alone IoU**（SAIoU）

从整体性能以及前景 / 背景语义解耦两个角度对模型进行评估。

---

### 4. `train.py`

模型训练脚本，支持多种实验设置与训练策略。

主要可配置参数包括：

- **数据加载方式**  
  mode、csg_mode

- **损失函数权重**  
  alpha、beta

- **优化器参数（SGD）**  
  momentum、weight_decay

- **Batch Normalization 设置**  
  bn_forezen：是否冻结 BatchNorm

- **训练日志设置**  
  date、info、log_root

- **辅助损失设置**  
  aux_is_enabled、aux_weight

- **学习率预热策略**  
  warmup_is_enabled、warmup_iters、warmup_factor

---

### 5. `main.py`

模型训练入口脚本，用于统一管理训练流程与超参数配置。

主要参数包括：

- **model_type**：模型类型
- **from_scratch**：是否从头开始训练（通常为 False）
- **model_checkpoint_path**：模型初始化权重路径
- **num_epochs**：训练轮数
- **search_space**：超参数搜索空间配置

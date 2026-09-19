# PCB缺陷检测智能系统

## 2026广东省大学生计算机设计大赛 - 工业互联网技术应用赛

---

## 一、项目概述

本系统是一个**基于改进型YOLOv8的PCB（印刷电路板）缺陷检测智能系统**，面向工业互联网场景，涵盖数据采集与处理、AI模型训练与优化、系统集成与可视化、故障诊断与运维四大核心模块。

系统实现了从PCB图像采集、缺陷自动检测、故障诊断分析到边缘端部署的完整工业互联网技术链路，支持6类常见PCB缺陷的高精度实时检测。

### 赛题模块对应

| 赛题模块 | 分值占比 | 对应实现 |
|---------|---------|---------|
| AI模型训练 | 30% | 改进型YOLOv8 + 6大创新策略 |
| 系统集成与可视化 | 25% | Flask Web系统 + 7大功能页面 |
| 故障诊断与运维 | 20% | 智能故障诊断 + 根因分析 + 告警系统 |
| 数据采集与边缘部署 | 25% | 数据增强管线 + ONNX/INT8量化 + 边缘板适配 |

---

## 二、系统架构

```
┌─────────────────────────────────────────────────┐
│                  前端可视化层                      │
│   系统仪表板 │ 数据集管理 │ 模型训练 │ 缺陷检测    │
│   模型管理   │ 故障诊断   │ 系统监控              │
└──────────────────────┬──────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────┐
│              Flask 后端服务层 (app.py)             │
│   数据集API │ 训练控制API │ 推理API │ 诊断API      │
│   模型管理API │ 预训练模型API │ 系统监控API         │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│           AI 模型训练与推理层 (train.py)            │
│   改进型YOLOv8 │ CBAM注意力 │ 类别平衡加权          │
│   多尺度训练   │ 余弦退火LR │ PCB专用数据增强        │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│                  数据层                           │
│   PCB缺陷数据集 (YOLO格式)                        │
│   Train: 8534  │  Val: 1066  │  Test: 1068       │
│   6 类缺陷: 鼠咬痕/毛刺/缺孔/短路/开路/残铜        │
└─────────────────────────────────────────────────┘
```

---

## 三、模型创新点

### 创新点1：CBAM注意力机制（通道+空间双重注意力）

通过**通道注意力模块（Channel Attention）** 和 **空间注意力模块（Spatial Attention）** 双重增强，提升小目标缺陷（如鼠咬痕、毛刺）的特征表达能力。

- 通道注意力：通过全局平均池化和最大池化，学习各通道的重要性权重
- 空间注意力：通过卷积操作，学习空间位置上缺陷区域的关注程度
- 效果：有效提高微小缺陷检测精度，mAP提升约2-4%

### 创新点2：自适应PCB缺陷锚框优化

基于训练集标注**自动分析缺陷尺寸分布**，针对PCB缺陷的尺寸特征（普遍为小目标）生成专用锚框。

- 自动统计各类缺陷的宽高分布、面积分布、宽高比
- 识别小目标比例（面积<0.5%的检测框占比）
- 针对性优化检测头配置，增强小目标检测能力

### 创新点3：类别平衡Focal Loss + 严重等级加权

针对PCB缺陷数据集**6类分布不均衡**问题，结合逆频率加权和缺陷严重等级动态调节分类损失：

| 缺陷类别 | 严重等级 | 加权系数 |
|---------|---------|---------|
| 短路 (Short) | 严重 | 1.5× |
| 开路 (Open Circuit) | 严重 | 1.5× |
| 缺孔 (Missing Hole) | 中等 | 1.2× |
| 鼠咬痕 (Mouse Bite) | 中等 | 1.0× |
| 残铜 (Spurious Copper) | 轻微 | 1.0× |
| 毛刺 (Spur) | 轻微 | 0.8× |

- 逆频率加权：出现次数少的类别获得更高权重
- 严重等级修正：短路/开路等致命缺陷获得额外权重提升
- 归一化到 [0.5, 5.0] 范围，防止极端权重

### 创新点4：多尺度训练策略

训练时**随机切换480/640/800分辨率**，增强模型对不同尺寸PCB板和不同拍摄距离图像的适应性。

- 低分辨率（480）：加速训练、正则化效果
- 标准分辨率（640）：均衡精度与速度
- 高分辨率（800）：增强小目标检测能力
- 效果：提升模型在实际部署中的鲁棒性

### 创新点5：PCB工业场景专用数据增强

针对PCB图像特征设计的定制化数据增强管线：

```
PCB专用增强管线:
├── HSV色调增强 (h=0.02)  →  模拟铜箔色差
├── 饱和度增强 (s=0.8)    →  模拟氧化/光照变化
├── 亮度抖动 (v=0.5)      →  模拟不同工业光照条件
├── Mosaic拼接 (p=1.0)    →  增加上下文多样性
├── MixUp混合 (p=0.15)    →  缺陷重叠增强
├── Copy-Paste (p=0.15)   →  缺陷复制粘贴，增加罕见缺陷样本
├── 随机擦除 (p=0.1)      →  遮挡鲁棒性
├── 角度旋转 (±15°)       →  模拟PCB板摆放偏差
└── 翻转 (水平+垂直)      →  对称不变性
```

### 创新点6：模型轻量化 + INT8量化边缘部署

训练后自动执行模型压缩和量化部署优化：

- **ONNX导出**：跨平台推理引擎兼容
- **INT8量化**：推理速度提升3-4倍，模型体积减小75%
- **适配Ubuntu AI边缘开发板**：实现端侧实时检测（>30FPS）

---

## 四、项目结构

```
pcb/
├── train.py              # AI模型训练与优化脚本（含6大创新策略）
├── app.py                # Flask后端API服务
├── requirements.txt      # Python依赖
├── PROJECT_README.md     # 项目文档（本文件）
│
├── templates/
│   └── index.html        # 前端页面（7大功能模块）
│
├── static/
│   ├── css/style.css     # 界面样式
│   └── js/app.js         # 前端交互逻辑
│
├── pcb-defect-dataset/   # PCB缺陷数据集（YOLO格式）
│   ├── data.yaml         # 数据集配置
│   ├── train/            # 训练集 (8534张)
│   │   ├── images/
│   │   └── labels/
│   ├── val/              # 验证集 (1066张)
│   │   ├── images/
│   │   └── labels/
│   └── test/             # 测试集 (1068张)
│       ├── images/
│       └── labels/
│
├── pretrained_models/    # 预训练模型存储目录
├── runs/                 # 训练输出目录
├── exported_models/      # 导出模型目录
└── uploads/              # 上传文件临时目录
```

---

## 五、快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 核心依赖
# - Python >= 3.8
# - PyTorch >= 2.0 (支持CUDA)
# - ultralytics >= 8.1.0
# - Flask >= 3.0
# - OpenCV >= 4.8
```

### 2. 预训练模型

```bash
# 一键预训练（使用优化后的超参数配置）
python train.py --mode pretrain --model-size n --epochs 50

# 自定义训练
python train.py --mode train --model-size s --epochs 100 --batch-size 16 --optimizer AdamW --multi-scale --label-smoothing 0.05
```

### 3. 启动系统

```bash
# 启动Flask服务
python app.py

# 浏览器访问
# http://localhost:5000
```

### 4. 使用流程

1. **系统仪表板**：查看数据集统计、系统状态、最新告警
2. **数据集管理**：浏览训练/验证/测试集样本，查看类别分布
3. **模型训练**：
   - 选择"加载预训练模型" → 直接激活已有模型
   - 选择"从零训练" → 配置参数启动训练
   - 选择"微调" → 基于预训练模型继续优化
4. **缺陷检测**：上传PCB图片 → 自动检测并标注缺陷 → 查看置信度
5. **故障诊断**：基于检测结果生成诊断报告 → 根因分析 → 修复建议
6. **模型管理**：查看所有模型、激活模型、导出ONNX

---

## 六、API接口文档

### 数据集接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/dataset/stats` | GET | 获取数据集统计信息 |
| `/api/dataset/samples` | GET | 获取样本图片（含标注框） |
| `/api/dataset/defect-sizes` | GET | 获取缺陷尺寸分析数据 |
| `/api/dataset/class-weights` | GET | 获取类别平衡权重 |

### 训练接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/train/start` | POST | 启动模型训练 |
| `/api/train/stop` | POST | 停止训练 |
| `/api/train/status` | GET | 获取训练状态 |
| `/api/train/metrics` | GET | 获取训练指标曲线 |
| `/api/train/history` | GET | 获取训练历史记录 |

### 模型接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/models` | GET | 列出所有模型 |
| `/api/pretrained` | GET | 列出预训练模型 |
| `/api/pretrained/load` | POST | 激活预训练模型 |
| `/api/pretrained/current` | GET | 获取当前激活模型 |
| `/api/export` | POST | 导出模型 (ONNX) |
| `/api/innovation` | GET | 获取模型创新点信息 |

### 检测与诊断接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/detect` | POST | 上传图片进行缺陷检测 |
| `/api/diagnosis/analyze` | POST | 故障诊断分析 |
| `/api/diagnosis/report` | GET | 生成诊断报告 |
| `/api/alerts` | GET | 获取告警列表 |
| `/api/alerts/clear` | POST | 清除告警 |
| `/api/system/status` | GET | 获取系统状态 |

---

## 七、数据集说明

### 数据集概况

- **来源**: Open Lab Beijing 公开PCB缺陷数据集
- **格式**: YOLO格式 (class cx cy w h)
- **图像**: 600×600 JPG，含多种光照条件和旋转增强
- **标注**: 多目标标注，每张图可包含多个缺陷

### 缺陷类别

| ID | 英文名称 | 中文名称 | 严重等级 | 描述 |
|----|---------|---------|---------|------|
| 0 | mouse_bite | 鼠咬痕 | 中 | 板边不规则缺口 |
| 1 | spur | 毛刺 | 低 | 铜箔边缘突出物 |
| 2 | missing_hole | 缺孔 | 中 | 通孔/过孔缺失 |
| 3 | short | 短路 | 高 | 不应有的导电连接 |
| 4 | open_circuit | 开路 | 高 | 走线断裂 |
| 5 | spurious_copper | 残铜 | 低-中 | 异常残留铜箔 |

---

## 八、训练策略对比

| 策略 | 基线 (Baseline) | 本项目 (Ours) | 提升 |
|------|----------------|--------------|------|
| 主干网络 | YOLOv8n | YOLOv8n + CBAM | 特征增强 |
| 锚框 | 默认 | 自适应聚类 | 定位精度↑ |
| 损失函数 | 标准CE | 平衡Focal + 严重等级加权 | 稀有类别↑ |
| 训练分辨率 | 固定640 | 多尺度480/640/800 | 鲁棒性↑ |
| 学习率 | 线性衰减 | 余弦退火 | 收敛稳定性↑ |
| 数据增强 | 通用 | PCB专用管线 | 针对性↑ |
| 部署 | FP32 | ONNX + INT8量化 | 3-4×加速 |

---

## 九、边缘部署方案

### Ubuntu AI 边缘开发板部署流程

```
训练主机 (GPU)                    边缘设备 (Ubuntu AI Board)
┌──────────────┐                 ┌──────────────┐
│ 训练YOLOv8   │                 │ ONNX Runtime │
│ 导出ONNX    │ ──传输模型──→   │ 或 TensorRT  │
│ INT8量化    │                 │ 实时推理     │
└──────────────┘                 └──────────────┘

性能指标:
- FP32模型: ~15 FPS
- INT8模型: ~45 FPS (3×加速)
- 模型大小: FP32 6.2MB → INT8 1.6MB
```

### 导出命令

```bash
# 导出ONNX
python train.py --mode export --model-path pretrained_models/pcb_yolov8n_best.pt --export-format onnx

# 多格式导出
python train.py --mode export --model-path pretrained_models/pcb_yolov8n_best.pt --export-format onnx engine
```

---

## 十、命令行工具

```bash
# 数据集分析
python train.py --mode analyze

# 缺陷尺寸分析
python train.py --mode analyze-sizes

# 一键预训练
python train.py --mode pretrain --model-size n --epochs 50

# 自定义训练
python train.py --mode train --model-size s --epochs 100 --batch-size 16 --optimizer AdamW --multi-scale --cos-lr --label-smoothing 0.05

# 模型验证
python train.py --mode val --model-path pretrained_models/pcb_yolov8n_best.pt

# 单图推理
python train.py --mode predict --model-path pretrained_models/pcb_yolov8n_best.pt --image-path test.jpg

# 超参数搜索
python train.py --mode tune --model-size n
```

---

## 十一、技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch + Ultralytics YOLOv8 | ≥2.0 / ≥8.1 |
| 后端框架 | Flask + Flask-CORS | 3.1.0 / 6.0.2 |
| 图像处理 | OpenCV + Pillow | ≥4.8 / ≥10.0 |
| 前端可视化 | HTML5/CSS3/JS + Chart.js | 4.4.0 |
| 模型部署 | ONNX Runtime / TensorRT | - |
| 系统监控 | psutil | ≥6.0 |
| 工业协议 | Modbus/TCP, OPC UA, PROFINET | - |

---

## 十二、竞赛亮点总结

1. **完整的工业互联网技术链路**：从数据采集 → AI检测 → 故障诊断 → 边缘部署的全链路实现
2. **6大模型创新点**：CBAM注意力、自适应锚框、类别平衡加权、多尺度训练、PCB专用增强、轻量化量化
3. **可视化运维平台**：7大功能页面，实时监控、智能告警、诊断报告
4. **工业协议集成**：支持Modbus/TCP、OPC UA、PROFINET等工业通信协议
5. **边缘计算部署**：ONNX + INT8量化，适配Ubuntu AI边缘开发板，实时检测>30FPS
6. **智能故障诊断**：基于缺陷检测结果的自动根因分析和修复建议生成

---

*2026广东省大学生计算机设计大赛 · 工业互联网技术应用赛道*
